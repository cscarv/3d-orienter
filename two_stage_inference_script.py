import torch
import trimesh
import numpy as np
from pl_models.OrienterTrainerModel import OrienterTrainerModel
from pl_models.FlipperTrainerModel import FlipperTrainerModel
from utils.helpers import get_timestamp
from utils.inference_helpers import normalize_mesh, calibrate_conformal_flipper, conformal_flipper, flipper_voting_scheme, orient
from ml_models.orienter_model.DGCNNOrienter import DGCNNOrienter
from ml_models.orienter_model.DGCNNFlipper import DGCNNFlipper
import argparse 
import os
import json5
import time
from tqdm import tqdm

def main():
    # parse and load specs
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", "-e", default="config/default", help="Path to specs.json5")
    parser.add_argument("--slurm_id", "-s", default=get_timestamp(), help="Path to specs.json5")
    # parser.add_argument("--train_index_file", "-ti", default="data/shapenet_index_files/all_point_clouds/train.txt", help="Path to train index file")
    # parser.add_argument("--val_index_file", "-vi", default="data/shapenet_index_files/all_point_clouds/val.txt", help="Path to val index file")
    parser.add_argument("--inference_index_file", "-ii", default="data/objaverse_index_files/meshes/inference.txt", help="Path to inference index file")
    parser.add_argument("--orienter_ckpt_path", "-ock", default="pretrained_ckpts/orienter.ckpt", help="Path of checkpoint storing the model")
    parser.add_argument("--flipper_ckpt_path", "-fck", default="pretrained_ckpts/flipper.ckpt", help="Path of checkpoint storing the model")
    parser.add_argument("--results_dir", "-rd", default="results/inference_results_full_pipeline", help="Path to save the inference meshes")
    parser.add_argument("--num_candidates", "-nc", type=int, default=20, help="Number of candidates to consider in the voting scheme")
    parser.add_argument("--conformal", "-c", action='store_true', help="Use conformal prediction for flipper")
    parser.add_argument("--calibration_index_file", "-ci", default="data/shapenet_index_files/all_meshes/calibration.txt", help="Path to calibration index file")
    parser.add_argument("--flipper_voting_scheme", "-fv", action='store_true', help="Use voting scheme for flipper")
    args = parser.parse_args()    
    slurm_id = args.slurm_id
    exp_dir = args.exp_dir.rstrip(" /")
    train_index_file = args.inference_index_file # This can be a dummy file at inference time
    val_index_file = args.inference_index_file
    inference_index_file = args.inference_index_file
    orienter_ckpt_path = args.orienter_ckpt_path
    flipper_ckpt_path = args.flipper_ckpt_path
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    num_candidates = args.num_candidates
    conformal = args.conformal
    calibration_index_file = args.calibration_index_file
    use_flipper_voting_scheme = args.flipper_voting_scheme
    print(f"{exp_dir=}")
    print(f"{slurm_id=}") 
    print(f"{train_index_file=}")
    print(f"{val_index_file=}")
    print(f"{inference_index_file=}")
    print(f"{orienter_ckpt_path=}")
    print(f"{flipper_ckpt_path=}")
    print(f"{results_dir=}")
    print(f"{num_candidates=}")
    print(f"{conformal=}")
    print(f"{calibration_index_file=}")
    print(f"{use_flipper_voting_scheme=}")
    with open(os.path.join(exp_dir, "specs.json5"), "r") as file:
        specs = json5.load(file)
    specs["exp_dir"] = exp_dir

    # Load flip matrices
    flip_matrices = torch.load("utils/24_cube_flips.pt").cpu().numpy()

    # path to index file for Shapenet inference meshes to pass to TrainerModels
    # shapenet_inference_index_file = "data/shapenet_index_files/all_meshes/100_inference.txt"

    # Load orienter trainer from checkpoint
    dgcnn_args = argparse.Namespace()
    dgcnn_args.k = 20
    dgcnn_args.emb_dims = 1024
    dgcnn_args.dropout = 0.5
    core_model = DGCNNOrienter(dgcnn_args, rotation_representation="procrustes")

    # Load orienter model from checkpoint
    trainer_module = OrienterTrainerModel.load_from_checkpoint(orienter_ckpt_path,
                                                               specs = specs,
                                                               core_model = core_model, 
                                                               train_loss_fn = 'octahedral_invariant',
                                                               rotation_representation = "6d",
                                                               train_index_file_path = train_index_file,
                                                               val_index_file_path = val_index_file,
                                                               inference_index_file_path = inference_index_file,
                                                               preload = False,
                                                               num_points_per_cloud = 2000,
                                                               train_batch_size = 48,
                                                               val_batch_size = 48,
                                                               unlock_every_k_epochs = 10,
                                                               start_lr = 1e-4
                                                               )
    orienter_model = trainer_module.model
    orienter_model.eval()

    # Load flipper trainer from checkpoint
    dgcnn_args = argparse.Namespace()
    dgcnn_args.k = 20
    dgcnn_args.emb_dims = 1024
    dgcnn_args.dropout = 0.5
    core_model = DGCNNFlipper(dgcnn_args, output_channels=24)

    # Load flipper model from checkpoint
    trainer_module = FlipperTrainerModel.load_from_checkpoint(flipper_ckpt_path,
                                                              specs = specs,
                                                              core_model = core_model,
                                                              train_index_file_path = train_index_file,
                                                              val_index_file_path = val_index_file,
                                                              inference_index_file_path = inference_index_file,
                                                              preload = False,
                                                              confusion_matrices = False,
                                                              up_flipper = False,
                                                              num_points_per_cloud = 2000,
                                                              train_batch_size = 48,
                                                              val_batch_size = 48,
                                                              unlock_every_k_epochs = 10,
                                                              start_lr = 1e-4
                                                              )
    flipper_model = trainer_module.model
    flipper_model.eval()

    # Load inference meshes
    with open(inference_index_file, 'r') as f:
        inference_mesh_paths = f.readlines()

    # Calibrate the flipper
    if conformal:
        print("Calibrating conformal flipper...")
        calibration_constant = calibrate_conformal_flipper(flipper_model, calibration_index_file, flip_matrices, confidence_level=0.5)
        print(f"Calibration constant: {calibration_constant}")
    
    # iterate through inference meshes and run the orienter and flipper models

    start_time = time.time()

    for inference_mesh_path in tqdm(inference_mesh_paths):
        inference_mesh_path = inference_mesh_path.strip()
        try:
            inference_tmesh = trimesh.load(inference_mesh_path, force="mesh")
        except:
            print(f"Error loading mesh: {inference_mesh_path}")
            continue
        inference_verts, inference_faces = normalize_mesh(inference_tmesh.vertices, inference_tmesh.faces)
        mesh = trimesh.Trimesh(inference_verts, inference_faces)

        # randomly rotate the mesh for test purposes
        test_rotation = trimesh.transformations.random_rotation_matrix()
        # Apply the rotation to the mesh
        mesh.apply_transform(test_rotation)

        # Run orienter model
        feats_oriented = orient(mesh, orienter_model, num_candidates=num_candidates)

        # Run flipper model
        if conformal:
            prediction_set = conformal_flipper(flipper_model, feats_oriented, calibration_constant)
            relevant_flip_matrices = flip_matrices[prediction_set] # (num_relevant_flip_matrices, 3, 3)
            relevant_flip_matrices = relevant_flip_matrices.transpose(0, 2, 1) # (num_relevant_flip_matrices, 3, 3)
            # Apply all relevant flip matrices to copies of the mesh and save them
            for i, flip_matrix in enumerate(relevant_flip_matrices):
                # Convert to 4x4 homogeneous matrix
                flip_matrix = np.pad(flip_matrix, ((0, 1), (0, 1)), mode='constant')
                flip_matrix[-1, -1] = 1
                # Apply the flip matrix to the mesh
                mesh_copy = mesh.copy()
                mesh_copy.apply_transform(flip_matrix)
                # Save the mesh
                results_subdir = os.path.join(results_dir, inference_mesh_path.split("/")[-3])
                os.makedirs(results_subdir, exist_ok=True)
                mesh_copy.export(os.path.join(results_subdir, f"{inference_mesh_path.split('/')[-1].split('.')[0]}_{i}.obj"))

        elif use_flipper_voting_scheme:
            flip_matrix = flipper_voting_scheme(flipper_model, feats_oriented, flip_matrices, num_candidates=50).cpu().numpy().T
            # Convert to 4x4 homogeneous matrix
            flip_matrix = np.pad(flip_matrix, ((0, 1), (0, 1)), mode='constant')
            flip_matrix[-1, -1] = 1
            # Apply the flip matrix to the mesh
            mesh.apply_transform(flip_matrix)
            # Save the mesh as an obj file
            results_subdir = os.path.join(results_dir, inference_mesh_path.split("/")[-3])
            os.makedirs(results_subdir, exist_ok=True)
            mesh.export(os.path.join(results_subdir, f"{inference_mesh_path.split('/')[-1].split('.')[0]}.obj"))


        else:
            with torch.no_grad():
                logits = flipper_model(feats_oriented)
            # Find the flip matrix with the highest score
            pred_index = torch.argmax(logits, dim=1).squeeze()
            flip_matrix = flip_matrices[pred_index].T
            # Convert to 4x4 homogeneous matrix
            flip_matrix = np.pad(flip_matrix, ((0, 1), (0, 1)), mode='constant')
            flip_matrix[-1, -1] = 1
            # Apply the flip matrix to the mesh
            mesh.apply_transform(flip_matrix)
            # Save the mesh as an obj file
            results_subdir = os.path.join(results_dir, inference_mesh_path.split("/")[-3])
            os.makedirs(results_subdir, exist_ok=True)
            mesh.export(os.path.join(results_subdir, f"{inference_mesh_path.split('/')[-1].split('.')[0]}.obj"))

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()