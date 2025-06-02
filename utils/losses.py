import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

cube_flips = torch.load("utils/24_cube_flips.pt")

def batch_pairwise_dist(x,y):
	bs, num_points_x, points_dim = x.size()
	_, num_points_y, _ = y.size()
	xx = torch.bmm(x, x.transpose(2,1))
	yy = torch.bmm(y, y.transpose(2,1))
	zz = torch.bmm(x, y.transpose(2,1))
	diag_ind_x = torch.arange(0, num_points_x, device=x.device) # send to same device as x
	diag_ind_y = torch.arange(0, num_points_y, device=y.device) # send to same device as y
	rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2,1))
	ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
	P = (rx.transpose(2,1) + ry - 2*zz)
	return P

def chamfer_distance(preds, gts):
	P = batch_pairwise_dist(gts, preds)
	mins, _ = torch.min(P, 1)
	loss_1 = torch.sum(mins, dim=1) # (B,)
	mins, _ = torch.min(P, 2)
	loss_2 = torch.sum(mins, dim=1) # (B,)
	return loss_1 + loss_2 # (B,)

def l2_loss(up_predicted, front_predicted, target_rotation_matrices):
	"""Extract the up and front vectors from the rotation matrices and compute the L2 loss."""
	# target up-vector is the second column of the target rotation matrix
	up_target = target_rotation_matrices[:, :, 1] # (B, 3)
	up_loss = torch.sum((up_predicted - up_target) ** 2, dim=1) # (B,)

	# target front-vector is the third column of the target rotation matrix
	front_target = target_rotation_matrices[:, :, 2] # B x 3
	front_loss = torch.sum((front_predicted - front_target) ** 2, dim=1) # (B,)

	return up_loss, front_loss

def abs_cos_loss(up_predicted, front_predicted, target_rotation_matrices):
	"""Extract the up and front vectors from the rotation matrices and compute 1 - absolute cosine loss."""
	# target up-vector is the second column of the target rotation matrix
	up_target = target_rotation_matrices[...,1] # (B, 3)
	up_loss = 1 - torch.abs(F.cosine_similarity(up_predicted, up_target, dim=-1)) # (B,)

	# target front-vector is the third column of the target rotation matrix
	front_target = target_rotation_matrices[...,2] # B x 3
	front_loss = 1 - torch.abs(F.cosine_similarity(front_predicted, front_target, dim=-1)) # (B,)

	return up_loss, front_loss

def octahedral_invariant_loss(up_predicted, front_predicted, target_rotation_matrices):
	"""Compute a loss that is invariant to the octahedral symmetries of the rotation matrices."""
	# compute abs cos loss for all 6 possible permutations of columns of the target_rotation_matrices
	# the best match is the one that minimizes the loss

	perm_list = [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]
	losses = []
	for perm in perm_list:
		target_rotation_matrices_perm = target_rotation_matrices[...,perm]
		perm_up_loss, perm_front_loss = abs_cos_loss(up_predicted, front_predicted, target_rotation_matrices_perm)
		losses.append(perm_up_loss + perm_front_loss)
	losses = torch.stack(losses, dim=-1) # (B, 6)
	loss, _ = torch.min(losses, dim=-1) # (B,)

	return loss

def quotient_regression_loss(predicted_rotation_matrices, target_rotation_matrices):
	"""Compute the L2 loss quotiented by the octahedral group."""
	# compute the L2 loss for all 24 possible flips of the predicted rotation matrices
	# the best match is the one that minimizes the loss
	
	losses = []
	for flip_matrix in cube_flips:
		flip_matrix = flip_matrix.to(predicted_rotation_matrices) # (3, 3)
		flip_matrices = flip_matrix.expand_as(predicted_rotation_matrices) # (B, 3, 3)
		flipped_target_rotation_matrices = torch.bmm(target_rotation_matrices, flip_matrices) # (B, 3, 3)
		loss = torch.sum((predicted_rotation_matrices - flipped_target_rotation_matrices) ** 2, dim=(1,2)) # (B,)
		losses.append(loss)
	losses = torch.stack(losses, dim=-1) # (B, 24)
	loss, _ = torch.min(losses, dim=-1) # (B,)

	return loss

def nuclear_norm_loss(predicted_rotation_matrices, target_rotation_matrices):
	"""Compute the nuclear norm loss between the predicted and target rotation matrices."""
	# compute the nuclear norm of the difference between the predicted and target rotation matrices
	loss = torch.linalg.matrix_norm(predicted_rotation_matrices - target_rotation_matrices, ord='nuc', dim=(1,2)) # (B,)

	return loss

def up_flipper_loss(logits, flip_indices, up_equivalent_flip_dict):
	"""Compute the loss for the up-flipper network."""
	# for each flip index, extract the list of equivalent flips from the up_equivalent_flip_list
	# compute the loss for each equivalent flip and take the minimum
	# make it a batched operation
	equivalent_indices_tensor = torch.tensor([list(up_equivalent_flip_dict[int(flip_index.cpu())]) for flip_index in flip_indices]).to(logits) # (B, 4)
	# iterate through the columns of equivalent_indices_tensor and take the minimum cross-entropy loss
	losses = []
	for i in range(4):
		equivalent_indices = equivalent_indices_tensor[:,i].long()
		loss = F.cross_entropy(logits, equivalent_indices)
		losses.append(loss)
	losses = torch.stack(losses, dim=-1) # (B, 4)
	loss, _ = torch.min(losses, dim=-1) # (B,)

	return loss

def full_rotation_angular_error(predicted_rotation_matrices, target_rotation_matrices):
	"""Compute the angular error between the predicted and target rotation matrices."""
	# first compute difference rotation matrices
	difference_rotation_matrices = torch.bmm(predicted_rotation_matrices, target_rotation_matrices.transpose(2,1)) # (B, 3, 3)
	# compute the trace of the difference rotation matrices
	traces = torch.diagonal(difference_rotation_matrices, dim1=1, dim2=2).sum(dim=1) # (B,)
	# compute the angular error
	angular_errors = torch.acos((traces - 1) / 2) # (B,)
	# convert to degrees
	angular_errors = angular_errors * 180 / np.pi

	return angular_errors