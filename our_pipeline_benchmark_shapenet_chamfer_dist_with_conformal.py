import argparse
import os
import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import trimesh
from dataloaders.MultiMeshDataset import MultiMeshDataset
from pl_models.OrienterTrainerModel import OrienterTrainerModel
from pl_models.FlipperTrainerModel import FlipperTrainerModel
from ml_models.orienter_model.DGCNNOrienter import DGCNNOrienter
from ml_models.orienter_model.DGCNNFlipper import DGCNNFlipper
from utils.helpers import get_timestamp, rotation_from_model_outs
from utils.inference_helpers import voting_scheme, calibrate_conformal_flipper, conformal_flipper
from utils.losses import chamfer_distance
import json5

torch.cuda.current_device()
torch.cuda._initialized = True
# torch.multiprocessing.set_sharing_strategy('file_system') # to avoid "too many open files" error

def compute_losses(batch, orienter_model, flipper_model, flip_matrices, calibration_constant):
    """Mimics validation step from our orienter-3d."""
    data_indices, xyzs_rotated, xyz, target_rotation_matrices, normals_rotated = batch
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
    prediction_set = conformal_flipper(flipper_model, feats_oriented, calibration_constant)
    relevant_flip_matrices = flip_matrices[prediction_set] # (num_relevant_flip_matrices, 3, 3)
    aps_size = relevant_flip_matrices.shape[0]
    chamfer_dist_list = []
    xyzs_oriented_flipped_list = []
    for flip_matrix in relevant_flip_matrices:
        xyzs_oriented_flipped = xyzs_oriented.cpu().numpy() @ flip_matrix
        # Compute the chamfer distance between xyzs_oriented_flipped and xyz
        chamfer_dist = chamfer_distance(torch.from_numpy(xyzs_oriented_flipped).to(xyz).unsqueeze(0), xyz).squeeze()
        chamfer_dist_list.append(chamfer_dist)
        xyzs_oriented_flipped_list.append(xyzs_oriented_flipped)
    # Find index of flip matrix with lowest chamfer distance
    chamfer_dist_list = torch.stack(chamfer_dist_list, dim=0) # (num_relevant_flip_matrices)
    best_flip_matrix_index = torch.argmin(chamfer_dist_list)
    best_xyzs_oriented_flipped = xyzs_oriented_flipped_list[best_flip_matrix_index]
    best_chamfer_dist = chamfer_dist_list[best_flip_matrix_index]

    # Divide by the size of the point cloud
    best_chamfer_dist /= xyz.shape[1] 

    return best_chamfer_dist, best_xyzs_oriented_flipped, target_rotation_matrices, aps_size, data_indices

def main():
    # parse and load specs
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", "-e", default="config/default", help="Path to specs.json5")
    parser.add_argument("--slurm_id", "-s", default=get_timestamp(), help="Path to specs.json5")
    parser.add_argument("--train_index_file", "-ti", default="data/sample_index.txt", help="Path to train index file")
    parser.add_argument("--val_index_file", "-vi", default="data/sample_index.txt", help="Path to val index file")
    parser.add_argument("--inference_index_file", "-ii", default="data/sample_index.txt", help="Path to inference index file")
    parser.add_argument("--calibration_index_file", "-ci", default="data/sample_index.txt", help="Path to calibration index file")
    parser.add_argument("--all_index_file", "-ai", default="data/sample_index.txt", help="Path to index file for all point clouds")
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
    calibration_index_file = args.calibration_index_file
    all_index_file_path = args.all_index_file
    preload = args.preload
    orienter_ckpt_path = args.orienter_ckpt_path
    flipper_ckpt_path = args.flipper_ckpt_path
    use_flipper_voting_scheme = args.flipper_voting_scheme
    top4 = args.top4
    conformal = True # fix the value of conformal to True
    print(f"{exp_dir=}")
    print(f"{slurm_id=}") 
    print(f"{train_index_file_path=}")
    print(f"{val_index_file_path=}")
    print(f"{inference_index_file_path=}")
    print(f"{calibration_index_file=}")
    print(f"{all_index_file_path=}")
    print(f"{preload=}")
    print(f"{orienter_ckpt_path=}")
    print(f"{flipper_ckpt_path=}")
    print(f"{use_flipper_voting_scheme=}")
    print(f"{top4=}")
    print(f"{conformal=}")
    with open(os.path.join(exp_dir, "specs.json5"), "r") as file:
        specs = json5.load(file)
    specs["exp_dir"] = exp_dir

    # Load PL module from checkpoint
    dgcnn_args = argparse.Namespace()
    dgcnn_args.k = 20
    dgcnn_args.emb_dims = 1024
    dgcnn_args.dropout = 0.5
    core_model = DGCNNOrienter(dgcnn_args, rotation_representation="procrustes").cuda()

    val_dataloader = DataLoader(MultiMeshDataset(index_file_path = val_index_file_path, sample_size = 2000, preload=False, chamfer=True), 
                                                batch_size = 1,
                                                shuffle = False,
                                                num_workers = 0,
                                                persistent_workers = False
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

    # Calibrate the flipper
    if conformal:
        print("Calibrating conformal flipper...")
        calibration_constant = calibrate_conformal_flipper(flipper_model, calibration_index_file, flip_matrices, confidence_level=0.5)
        print(f"Calibration constant: {calibration_constant}")

    # record losses
    chamfer_dists = []

    # record aps sizes
    aps_sizes = []

    # record data indices
    data_indices_list = []

    results_dir = "benchmark_results/our_pipeline_chamfer_distances_conformal_pred"
    os.makedirs(results_dir, exist_ok=True)
    point_cloud_dir = "benchmark_results/our_pipeline_chamfer_distances_conformal_pred/point_clouds"
    os.makedirs(point_cloud_dir, exist_ok=True)

    for i, batch in enumerate(tqdm(val_dataloader)):
        with torch.no_grad():
            best_chamfer_dist, best_xyzs_oriented_flipped, target_rotation_matrices, aps_size, data_indices = compute_losses(batch, orienter_model, flipper_model, flip_matrices, calibration_constant)
            chamfer_dists.append(best_chamfer_dist)
            aps_sizes.append(aps_size)
            data_indices_list.append(data_indices)
            # Immediately save xyzs_oriented_flipped to point_cloud_dir as an obj file
            # make a mesh from xyzs_rotated
            point_cloud_oriented = trimesh.Trimesh(vertices=best_xyzs_oriented_flipped.squeeze(), faces=[])
            point_cloud_oriented.export(f"{point_cloud_dir}/point_cloud_{i}.obj")
            del point_cloud_oriented
        # print running average of chamfer_dists
        print(f"Mean best chamfer dist so far: {np.mean(chamfer_dists)}")
        # print running average of aps_sizes
        print(f"Mean aps size so far: {np.mean(aps_sizes)}")
        # print running median of aps_sizes
        print(f"Median aps size so far: {np.median(aps_sizes)}")
    
    chamfer_dists = torch.stack(chamfer_dists, dim=0)
    aps_sizes = torch.tensor(aps_sizes)
    data_indices_all = torch.cat(data_indices_list, dim=0)

    # Compute mean and std of chamfer dists
    chamfer_dists_mean = chamfer_dists.mean()
    chamfer_dists_std = chamfer_dists.std()

    print(f"Mean chamfer dist between predicted and GT shape: {chamfer_dists_mean}")
    print(f"Std of chamfer dist between predicted and GT shape: {chamfer_dists_std}")

    # Save losses
    np.save(os.path.join(results_dir, "best_chamfer_dists.npy"), chamfer_dists.cpu().numpy())
    np.save(os.path.join(results_dir, "aps_sizes.npy"), aps_sizes.cpu().numpy())
    np.save(os.path.join(results_dir, "data_indices_all.npy"), data_indices_all.cpu().numpy())

if __name__ == "__main__":
    main()