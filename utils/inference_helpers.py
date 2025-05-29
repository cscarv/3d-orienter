import os
import torch
import trimesh
import torch.nn.functional as F
import numpy as np
from utils.helpers import rotation_from_model_outs, random_rotation_matrix
from utils.losses import octahedral_invariant_loss

def normalize_mesh(verts, faces):
    # compute bounding box and center the verts
    min_xyz = verts.min(0)
    max_xyz = verts.max(0)
    center = (min_xyz + max_xyz) / 2
    verts = verts - center
    # fit into radius 1 sphere
    maxrad = np.sqrt((verts**2).sum(1)).max()
    verts = verts / maxrad
    verts = verts * 0.95
    return verts, faces

def calibrate_conformal_flipper(flipper_model, calibration_index_file, flip_matrices, confidence_level=0.9):
    # Returns the calibration constant
    # Assumes that the model is already trained
    # Load calibration samples
    with open(calibration_index_file, 'r') as f:
        calibration_mesh_paths = f.readlines()
    calibration_scores = []
    for calibration_mesh_path in calibration_mesh_paths:
        calibration_mesh_path = calibration_mesh_path.strip()
        calibration_tmesh = trimesh.load(calibration_mesh_path, force="mesh")
        calibration_verts, calibration_faces = normalize_mesh(calibration_tmesh.vertices, calibration_tmesh.faces)
        calibration_mesh = trimesh.Trimesh(calibration_verts, calibration_faces)
        # apply a random flip to the mesh
        random_flip_idx = np.random.randint(24)
        random_flip = flip_matrices[random_flip_idx]
        # Convert to 4x4 homogeneous matrix
        random_flip = np.pad(random_flip, ((0, 1), (0, 1)), mode='constant')
        random_flip[-1, -1] = 1
        # Apply the rotation to the mesh
        calibration_mesh.apply_transform(random_flip)
        # Obtain features
        xyzs, faces = calibration_mesh.sample(2000, return_index=True)
        normals = calibration_mesh.face_normals[faces]
        xyzs = torch.as_tensor(xyzs).to(next(flipper_model.parameters()))
        normals = torch.as_tensor(normals).to(next(flipper_model.parameters()))
        feats = torch.cat([xyzs, normals], dim=1).unsqueeze(0)
        # Calibrate the flipper
        with torch.no_grad():
            # Get the predictions
            calibration_logits = flipper_model(feats)
            calibration_probs = F.softmax(calibration_logits, dim=1)
            # To obtain calibration score, we sort the probs in descending order and sum until we reach the random_flip_idx
            sorted_probs, sorted_indices = torch.sort(calibration_probs, descending=True, dim=1)
            # Find the index of the random_flip_idx in the sorted_indices
            true_idx_position = torch.nonzero(sorted_indices.squeeze() == random_flip_idx).item()
            # Calibration score is sum of top model probs up to and including the true index
            calibration_score = sorted_probs.squeeze()[:true_idx_position+1].sum()
            calibration_scores.append(calibration_score)
    calibration_scores = torch.tensor(calibration_scores)
    calibration_constant = torch.quantile(calibration_scores, confidence_level)
    # save the calibration scores
    calibration_scores = calibration_scores.cpu().numpy()
    save_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'calibration_scores.npy')
    np.save(save_path, calibration_scores)
    return calibration_constant

def conformal_flipper(flipper_model, feats_oriented, quantile):
    with torch.no_grad():
        # Get the predictions
        logits = flipper_model(feats_oriented) # (1, 24)
        probs = F.softmax(logits, dim=1) # (1, 24)
        # Obtain the prediction set
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
        cumulative_prob = 0
        prediction_set = []
        for i, (prob, idx) in enumerate(zip(sorted_probs.squeeze(), sorted_indices.squeeze())):
            cumulative_prob += prob
            prediction_set.append(idx.item())
            if cumulative_prob >= quantile:
                break
    return torch.tensor(prediction_set, dtype=torch.long).cpu().numpy()

def flipper_voting_scheme(flipper_model, feats_oriented, cube_flip_matrices, num_candidates=5):
    # feats_oriented: (1, num_points, 6)

    # Convert cube_flip_matrices to tensor and move to the same device as feats_oriented if necessary
    if not isinstance(cube_flip_matrices, torch.Tensor):
        cube_flip_matrices = torch.as_tensor(cube_flip_matrices).to(feats_oriented)
    
    # Generate random flip matrices
    # Draw random indices in the range [0, 24)
    random_flip_indices = torch.randint(0, 24, (num_candidates,)).to(feats_oriented.device) # (num_candidates,)
    random_flips = cube_flip_matrices[random_flip_indices] # (num_candidates, 3, 3)
    
    # Apply random flips to feats_oriented
    feats_oriented = feats_oriented.repeat(num_candidates, 1, 1) # (num_candidates, num_points, 6)
    xyzs_oriented = feats_oriented[:, :, :3] # (num_candidates, num_points, 3)
    normals_oriented = feats_oriented[:, :, 3:] # (num_candidates, num_points, 3)
    xyzs_oriented_flipped = torch.bmm(xyzs_oriented, random_flips.transpose(1, 2)) # (num_candidates, num_points, 3)
    normals_oriented_flipped = torch.bmm(normals_oriented, random_flips.transpose(1, 2)) # (num_candidates, num_points, 3)
    feats_oriented_flipped = torch.cat([xyzs_oriented_flipped, normals_oriented_flipped], dim=2) # (num_candidates, num_points, 6)
    
    # Predict using the flipper model
    with torch.no_grad():
        flip_logits = flipper_model(feats_oriented_flipped) # (num_candidates, 24)
        flip_probs = F.softmax(flip_logits, dim=1) # (num_candidates, 24)
    
    # Take the argmax of the flip_probs along the last dimension to get the predicted flip
    predicted_flip_indices = torch.argmax(flip_probs, dim=1) # (num_candidates,) 
    predicted_flips = cube_flip_matrices[predicted_flip_indices] # (num_candidates, 3, 3)

    # Apply the inverse of random flips to the predictions
    predicted_flips_original_space = torch.bmm(random_flips.transpose(1, 2) ,predicted_flips) # (num_candidates, 3, 3)

    # Now vote for the most common flip among predicted_flips_original_space
    unique, counts = torch.unique(predicted_flips_original_space, return_counts=True, dim=0)
    winning_flip = unique[torch.argmax(counts)]

    return winning_flip # (3, 3)

def voting_scheme(xyzs, normals, orienter_model, num_candidates=5, return_min=False):
    # xyzs: (num_points, 3)
    # normals: (num_points, 3)

    rotation_representation = orienter_model.rotation_representation
    
    # Generate random rotation matrices
    random_rotations = torch.stack([random_rotation_matrix() for _ in range(num_candidates)]).to(xyzs) # (num_candidates, 3, 3)
    
    # Apply random rotations to xyzs and normals
    xyzs = xyzs.unsqueeze(0).repeat(num_candidates, 1, 1) # (num_candidates, num_points, 3)
    normals = normals.unsqueeze(0).repeat(num_candidates, 1, 1) # (num_candidates, num_points, 3)
    xyzs_rotated = torch.bmm(xyzs, random_rotations.transpose(1, 2)) # (num_candidates, num_points, 3)
    normals_rotated = torch.bmm(normals, random_rotations.transpose(1, 2)) # (num_candidates, num_points, 3)
    feats_rotated = torch.cat([xyzs_rotated, normals_rotated], dim=2) # (num_candidates, num_points, 6)
    
    # Predict using the orienter model
    with torch.no_grad():
        if rotation_representation == "6d":
            up_predicted, front_predicted = orienter_model(feats_rotated)
        elif rotation_representation == "procrustes":
            predicted_rotations = orienter_model(feats_rotated) # (B, 3, 3)
            up_predicted = predicted_rotations[:, :, 1] # (B, 3)
            front_predicted = predicted_rotations[:, :, 2] # (B, 3)
    
    # Apply the inverse of random rotations to the predictions
    up_predicted = torch.bmm(up_predicted.unsqueeze(1), random_rotations).squeeze() # (num_candidates, 3)
    front_predicted = torch.bmm(front_predicted.unsqueeze(1), random_rotations).squeeze() # (num_candidates, 3)
    
    # Compute rotation matrices from model outputs
    pred_rotation_matrices = rotation_from_model_outs(up_predicted, front_predicted) # (num_candidates, 3, 3)

    # Compute octahedral losses

    # First expand up_preds, front_preds to (num_candidates, num_candidates, 3) for pairwise comparison
    up_preds_i = up_predicted.unsqueeze(1).expand(-1, num_candidates, -1) # (num_candidates, num_candidates, 3)
    front_preds_i = front_predicted.unsqueeze(1).expand(-1, num_candidates, -1) # (num_candidates, num_candidates, 3)

    # Do the same for pred_rotation_matrices
    pred_rotation_matrices_j = pred_rotation_matrices.unsqueeze(0).expand(num_candidates, -1, -1, -1) # (num_candidates, num_candidates, 3, 3)

    # Compute octahedral losses
    pairwise_loss_matrix = octahedral_invariant_loss(up_preds_i, front_preds_i, pred_rotation_matrices_j) # (num_candidates, num_candidates)

    # Compute sum of losses for each candidate
    sum_losses = pairwise_loss_matrix.sum(dim=1) # (num_candidates)

    # Find the candidate with the minimum sum of losses
    winner_index = torch.argmin(sum_losses)
    up_winner = up_predicted[winner_index]
    front_winner = front_predicted[winner_index]

    if return_min:
        return up_winner, front_winner, torch.min(sum_losses)
    else:
        return up_winner, front_winner

def orient(mesh, orienter_model, num_candidates=5):
    # Sample points and normals from the mesh
    xyzs, faces = mesh.sample(2000, return_index=True)
    normals = mesh.face_normals[faces]
    xyzs = torch.as_tensor(xyzs).to(next(orienter_model.parameters()))
    normals = torch.as_tensor(normals).to(next(orienter_model.parameters()))
    # Run the voting scheme to generate the predicted rotation matrix
    up_winner, front_winner = voting_scheme(xyzs, normals, orienter_model, num_candidates=num_candidates) # (3,), (3,)
    up_winner = up_winner.unsqueeze(0)
    front_winner = front_winner.unsqueeze(0)
    # Force front_winner to be orthogonal to up_winner
    front_winner = front_winner - torch.sum(front_winner * up_winner, dim=1, keepdim=True) * up_winner
    # Normalize front_winner again
    front_winner = F.normalize(front_winner, p=2, dim=1)
    # Compute the rotation matrix from the predictions
    predicted_rotations = rotation_from_model_outs(up_winner, front_winner).squeeze()
    # Invert the predicted rotation matrix by taking the transpose
    inverse_predicted_rotation = (predicted_rotations.T).cpu().numpy()
    # Convert to 4x4 homogeneous matrix
    inverse_predicted_rotation = np.pad(inverse_predicted_rotation, ((0, 1), (0, 1)), mode='constant')
    inverse_predicted_rotation[-1, -1] = 1
    # Apply the predicted rotation to the mesh
    mesh.apply_transform(inverse_predicted_rotation)
    # also apply predicted_rotations to feats
    xyzs_oriented = xyzs.unsqueeze(0) @ predicted_rotations
    normals_oriented = normals.unsqueeze(0) @ predicted_rotations
    feats_oriented = torch.cat([xyzs_oriented, normals_oriented], dim=2)
    return feats_oriented