import argparse
import os
import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from dataloaders.MultiMeshDataset import MultiMeshDataset
from pl_models.OrienterTrainerModel import OrienterTrainerModel
from pl_models.FlipperTrainerModel import FlipperTrainerModel
from ml_models.orienter_model.DGCNNOrienter import DGCNNOrienter
from ml_models.orienter_model.DGCNNFlipper import DGCNNFlipper
from utils.helpers import get_timestamp, rotation_from_model_outs
from utils.inference_helpers import voting_scheme, flipper_voting_scheme
from utils.losses import abs_cos_loss
import json5

torch.cuda.current_device()
torch.cuda._initialized = True
torch.multiprocessing.set_sharing_strategy('file_system') # to avoid "too many open files" error

def octahedral_invariant_loss(up_predicted, front_predicted, target_rotation_matrices):
    """Compute a loss that is invariant to the octahedral symmetries of the rotation matrices."""
    # compute abs cos loss for all 6 possible permutations of columns of the target_rotation_matrices
    # the best match is the one that minimizes the loss
    perm_list = [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]
    up_losses = []
    front_losses = []
    for perm in perm_list:
        target_rotation_matrices_perm = target_rotation_matrices[...,perm]
        perm_up_loss, perm_front_loss = abs_cos_loss(up_predicted, front_predicted, target_rotation_matrices_perm)
        up_losses.append(perm_up_loss)
        front_losses.append(perm_front_loss)
    up_losses = torch.stack(up_losses, dim=-1) # (B, 6)
    up_loss, _ = torch.min(up_losses, dim=-1) # (B,)
    front_losses = torch.stack(front_losses, dim=-1) # (B, 6)
    front_loss, _ = torch.min(front_losses, dim=-1) # (B,)
    return up_loss, front_loss

def compute_losses(batch, orienter_model, flipper_model, flip_matrices, use_flipper_voting_scheme=False, top4=False):
    """Mimics validation step from our orienter-3d."""
    data_indices, xyzs_rotated, target_rotation_matrices, normals_rotated = batch
    # squeeze batch dimension and move to cuda
    xyzs_rotated = xyzs_rotated.squeeze().cuda()
    normals_rotated = normals_rotated.squeeze().cuda()
    target_rotation_matrices = target_rotation_matrices.cuda()
    # run voting scheme
    up_winner, front_winner = voting_scheme(xyzs_rotated.squeeze(), normals_rotated.squeeze(), orienter_model, num_candidates=50)
    # unsqueeze
    up_winner = up_winner.unsqueeze(0)
    front_winner = front_winner.unsqueeze(0)

    # Force front_winner to be orthogonal to up_winner
    front_winner = front_winner - torch.sum(front_winner * up_winner, dim=1, keepdim=True) * up_winner
    # Normalize front_winner again
    front_winner = F.normalize(front_winner, p=2, dim=1)
    # Compute the rotation matrix from the predictions
    predicted_rotations = rotation_from_model_outs(up_winner, front_winner).squeeze()
    # apply predicted_rotations to feats
    xyzs_oriented = xyzs_rotated @ predicted_rotations
    normals_oriented = normals_rotated @ predicted_rotations
    feats_oriented = torch.cat([xyzs_oriented, normals_oriented], dim=1).unsqueeze(0)

    # now apply flipper
    if use_flipper_voting_scheme:
        flip_matrix = flipper_voting_scheme(flipper_model, feats_oriented, flip_matrices, num_candidates=50).cpu().numpy()
    elif top4:
        with torch.no_grad():
            logits = flipper_model(feats_oriented)
        # Find the top 4 flip matrices
        _, top4_indices = torch.topk(logits, 4, dim=1)
        top4_indices = top4_indices.squeeze().cpu().numpy()
        flip_matrix = flip_matrices[top4_indices] # (4, 3, 3)
    else:
        with torch.no_grad():
            logits = flipper_model(feats_oriented)
        # Find the flip matrix with the highest score
        pred_index = torch.argmax(logits, dim=1).squeeze()
        flip_matrix = flip_matrices[pred_index]
    
    # compute predicted full rotation matrix -- composition of predicted_rotations and flip_matrix
    predicted_full_rotation = predicted_rotations.cpu().numpy() @ flip_matrix
    # up_target and front_target are the up and front vectors of the target rotation matrix
    if top4:
        up_predicted = torch.tensor(predicted_full_rotation[...,1])
        front_predicted = torch.tensor(predicted_full_rotation[...,2])
        # compute cos sim between up_predicted and up_target, front_predicted and front_target for all 4 possible target rotation matrices
        up_target = target_rotation_matrices[...,1].squeeze().repeat(4, 1)
        front_target = target_rotation_matrices[...,2].squeeze().repeat(4, 1)
        up_cos_sim = F.cosine_similarity(up_predicted.to(up_target), up_target, dim=-1) # (4)
        front_cos_sim = F.cosine_similarity(front_predicted.to(front_target), front_target, dim=-1) # (4)
        # take the max cos sim
        up_cos_sim, _ = torch.max(up_cos_sim, dim=0)
        front_cos_sim, _ = torch.max(front_cos_sim, dim=0)
    else:
        up_predicted = torch.tensor(predicted_full_rotation[...,1]).unsqueeze(0)
        front_predicted = torch.tensor(predicted_full_rotation[...,2]).unsqueeze(0)
        up_target = target_rotation_matrices[...,1]
        front_target = target_rotation_matrices[...,2]
        up_cos_sim = F.cosine_similarity(up_predicted.to(up_target), up_target) # (B,)
        front_cos_sim = F.cosine_similarity(front_predicted.to(front_target), front_target) # (B,)

    # apply flip_matrix to xyzs_oriented
    xyzs_oriented_flipped = xyzs_oriented.cpu().numpy() @ flip_matrix
    # print the mean cos sim
    print("Cos sim between up_winner and up_target:", up_cos_sim.squeeze().item())
    print("Cos sim between front_winner and front_target:", front_cos_sim.squeeze().item())

    return up_cos_sim, front_cos_sim, xyzs_oriented_flipped, target_rotation_matrices, data_indices

def main():
    # parse and load specs
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", "-e", default="config/default", help="Path to specs.json5")
    parser.add_argument("--slurm_id", "-s", default=get_timestamp(), help="Path to specs.json5")
    parser.add_argument("--train_index_file", "-ti", default="data/shapenet_index_files/all_point_clouds/train.txt", help="Path to train index file")
    parser.add_argument("--val_index_file", "-vi", default="data/shapenet_index_files/all_point_clouds/val.txt", help="Path to val index file")
    parser.add_argument("--inference_index_file", "-ii", default="data/shapenet_index_files/all_point_clouds/inference.txt", help="Path to inference index file")
    parser.add_argument("--all_index_file", "-ai", default="data/shapenet_index_files/all_point_clouds/all.txt", help="Path to index file for all point clouds")
    parser.add_argument("--preload", "-p", action='store_true', help="Preload meshes into memory at initialization")
    parser.add_argument("--orienter_ckpt_path", "-ck", default="pretrained_ckpts/orienter.ckpt", help="Path of checkpoint storing the orienter model")
    parser.add_argument("--flipper_ckpt_path", "-cf", default="pretrained_ckpts/flipper.ckpt", help="Path of checkpoint storing the flipper model")
    parser.add_argument("--flipper_voting_scheme", "-fv", action='store_true', help="Use voting scheme for flipper")
    parser.add_argument("--top4", "-t4", action='store_true', help="Use top 4 flip matrices")
    args = parser.parse_args()    
    slurm_id = args.slurm_id
    exp_dir = args.exp_dir.rstrip(" /")
    train_index_file_path = args.train_index_file
    val_index_file_path = args.val_index_file
    inference_index_file_path = args.inference_index_file
    all_index_file_path = args.all_index_file
    preload = args.preload
    orienter_ckpt_path = args.orienter_ckpt_path
    flipper_ckpt_path = args.flipper_ckpt_path
    use_flipper_voting_scheme = args.flipper_voting_scheme
    top4 = args.top4
    print(f"{exp_dir=}")
    print(f"{slurm_id=}") 
    print(f"{train_index_file_path=}")
    print(f"{val_index_file_path=}")
    print(f"{inference_index_file_path=}")
    print(f"{all_index_file_path=}")
    print(f"{preload=}")
    print(f"{orienter_ckpt_path=}")
    print(f"{flipper_ckpt_path=}")
    print(f"{use_flipper_voting_scheme=}")
    print(f"{top4=}")
    with open(os.path.join(exp_dir, "specs.json5"), "r") as file:
        specs = json5.load(file)
    specs["exp_dir"] = exp_dir

    # Load PL module from checkpoint
    dgcnn_args = argparse.Namespace()
    dgcnn_args.k = 20
    dgcnn_args.emb_dims = 1024
    dgcnn_args.dropout = 0.5
    core_model = DGCNNOrienter(dgcnn_args, rotation_representation="procrustes").cuda()

    val_dataloader = DataLoader(MultiMeshDataset(index_file_path = val_index_file_path, sample_size = 2000, preload=False), 
                                                batch_size = 1, # max we can handle
                                                shuffle = False,
                                                num_workers = 1,
                                                persistent_workers = True # else there's overhead on switch
                                                )
    
    # Load model from checkpoint
    orienter_trainer_module = OrienterTrainerModel.load_from_checkpoint(orienter_ckpt_path,
                                                               specs = specs,
                                                               core_model = core_model, 
                                                               train_loss_fn = "octahedral_invariant",
                                                               rotation_representation = "procrustes",
                                                               train_index_file_path = train_index_file_path,
                                                               val_index_file_path = val_index_file_path,
                                                               inference_index_file_path = inference_index_file_path,
                                                               preload = False,
                                                               num_points_per_cloud = 2000,
                                                               train_batch_size = 48,
                                                               val_batch_size = 48,
                                                               unlock_every_k_epochs = 10,
                                                               start_lr = 1e-4
                                                               )
    orienter_model = orienter_trainer_module.model.cuda()
    orienter_model.eval()

    # Load flip matrices
    flip_matrices = torch.load("utils/24_cube_flips.pt").cpu().numpy()

    # Load flipper trainer from checkpoint
    dgcnn_args = argparse.Namespace()
    dgcnn_args.k = 20
    dgcnn_args.emb_dims = 1024
    dgcnn_args.dropout = 0.5
    core_model = DGCNNFlipper(dgcnn_args, output_channels=24)

    # Load flipper model from checkpoint
    flipper_trainer_module = FlipperTrainerModel.load_from_checkpoint(flipper_ckpt_path,
                                                              specs = specs,
                                                              core_model = core_model,
                                                              train_index_file_path = train_index_file_path,
                                                              val_index_file_path = val_index_file_path,
                                                              inference_index_file_path = inference_index_file_path,
                                                              preload = False,
                                                              confusion_matrices = False,
                                                              up_flipper = False,
                                                              num_points_per_cloud = 2000,
                                                              train_batch_size = 48,
                                                              val_batch_size = 48,
                                                              unlock_every_k_epochs = 10,
                                                              start_lr = 1e-4
                                                              )
    flipper_model = flipper_trainer_module.model
    flipper_model.eval()

    # record losses
    up_cos_sims = []
    front_cos_sims = []

    # record point clouds and target rotation matrices
    xyzs_oriented_list = []
    target_rotation_matrices_list = []

    # record data indices
    data_indices_list = []

    for i, batch in enumerate(tqdm(val_dataloader)):
        with torch.no_grad():
            up_cos_sim, front_cos_sim, xyzs_oriented, target_rotation_matrices, data_indices = compute_losses(batch, orienter_model, flipper_model, flip_matrices, use_flipper_voting_scheme, top4)
            up_cos_sims.append(up_cos_sim)
            front_cos_sims.append(front_cos_sim)
            xyzs_oriented_list.append(torch.from_numpy(xyzs_oriented).unsqueeze(0))
            target_rotation_matrices_list.append(target_rotation_matrices)
            data_indices_list.append(data_indices)
        # print proportion of angular error < 10 degrees
        # angular error is < 10 degrees if cos sim is > 0.9848
        up_cos_sims_all = torch.stack(up_cos_sims, dim=0)
        print(f"Proportion of angular error < 10 degrees: {(up_cos_sims_all > 0.9848).float().mean()}")
    
    up_cos_sims = torch.stack(up_cos_sims, dim=0)
    front_cos_sims = torch.stack(front_cos_sims, dim=0)
    xyzs_oriented_all = torch.cat(xyzs_oriented_list, dim=0)
    target_rotation_matrices_all = torch.cat(target_rotation_matrices_list, dim=0)
    data_indices_all = torch.cat(data_indices_list, dim=0)

    # Compute mean and std of cos sims
    up_cos_sims_mean = up_cos_sims.mean()
    up_cos_sims_std = up_cos_sims.std()
    front_cos_sims_mean = front_cos_sims.mean()
    front_cos_sims_std = front_cos_sims.std()

    print(f"Mean cos sim between up_winner and up_target: {up_cos_sims_mean}")
    print(f"Std of cos sim between up_winner and up_target: {up_cos_sims_std}")
    print(f"Mean cos sim between front_winner and front_target: {front_cos_sims_mean}")
    print(f"Std of cos sim between front_winner and front_target: {front_cos_sims_std}")

    # Save losses, point clouds, rotation matrices
    results_dir = "benchmark_results/up_accuracy_benchmark_shapenet"
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, "up_cos_sims.npy"), up_cos_sims.cpu().numpy())
    np.save(os.path.join(results_dir, "front_cos_sims.npy"), front_cos_sims.cpu().numpy())
    np.save(os.path.join(results_dir, "xyzs_oriented_all.npy"), xyzs_oriented_all.cpu().numpy())
    np.save(os.path.join(results_dir, "target_rotation_matrices_all.npy"), target_rotation_matrices_all.cpu().numpy())
    np.save(os.path.join(results_dir, "data_indices_all.npy"), data_indices_all.cpu().numpy())

if __name__ == "__main__":
    main()