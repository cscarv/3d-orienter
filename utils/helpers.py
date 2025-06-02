import torch
import torch.nn.functional as F
import trimesh
from datetime import datetime
from scipy.spatial.transform import Rotation as R
import numpy as np
import random

def zero_out_nan_gradients(model):
    for param in model.parameters():
        if param.grad is not None:
            nan_mask = torch.isnan(param.grad)
            param.grad.data[nan_mask] = 0.0

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

import builtins
from contextlib import contextmanager
from tqdm import tqdm
@contextmanager
def redirect_print_to_tqdm():
    # only works with string printing. won't work with print(a,b,c,end=d)
    original_print = builtins.print
    try:
        # Override the print function with tqdm.write
        def tqdm_print(*args, **kwargs):
            tqdm.write(*args, **kwargs)

        builtins.print = tqdm_print
        yield
    finally:
        # Restore the original print function
        builtins.print = original_print

def is_method(object, method_name):
    return hasattr(object, method_name) and callable(getattr(object, method_name))


from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
def get_log_dir(logger):
    if isinstance(logger, TensorBoardLogger):
        return logger.log_dir
    elif isinstance(logger, WandbLogger):
        return logger.save_dir
    else:
        raise Exception("Unknown logger and log_dir")

def get_num_workers(cpus_per_gpu):
    num_workers = cpus_per_gpu - 1
    if num_workers < 0:
        num_workers = 0
    return num_workers

from pytorch_lightning.callbacks import TQDMProgressBar
class CustomProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__(refresh_rate=20)  

    def init_train_tqdm(self):
        # refresh only every k iters
        # super().__init__()
        bar = super().init_train_tqdm()
        # no progress bar
        bar.dynamic_ncols = False
        bar.ncols = 0
        return bar

def random_rotation_matrix():
    """Draws random quaternion and converts it to a 3x3 rotation matrix."""
    q = np.random.randn(4) # random quaternion
    r = R.from_quat(q) # generate rotation object from quaternion -- Scipy auto-normalizes the quat
    rotation_matrix = r.as_matrix() # convert to 3x3 rotation matrix
    return torch.from_numpy(rotation_matrix) # (3,3)

def small_angle_random_rotation_matrix(s):
    """Draws a random unit vector and a small angle in [0,s], and generates a rotation matrix around that vector by that angle."""
    axis = np.random.randn(3) # random unit vector
    axis = axis / np.linalg.norm(axis)
    angle = np.random.rand() * s # random angle in [0,s]
    rotation_matrix = R.from_rotvec(angle * axis).as_matrix() # generate rotation matrix from axis-angle representation
    return torch.from_numpy(rotation_matrix) # (3,3)

def rotation_from_model_outs(up_predicted, front_predicted):
    """Compute the rotation matrix from the predicted up and front vectors."""
    side_predicted = torch.cross(up_predicted, front_predicted, dim=1)
    predicted_rotations = torch.stack([side_predicted, up_predicted, front_predicted], dim=2)
    return predicted_rotations # (B, 3, 3)

def visualize_model_on_mesh(model, mesh):
    """Visualize the model's action on an inference mesh."""
    # Sample a random rotation matrix using trimesh.transformations.random_rotation_matrix
    random_rotation = trimesh.transformations.random_rotation_matrix()
    # Apply the rotation to the mesh
    mesh.apply_transform(random_rotation)
    # Sample points from the mesh
    xyzs_rotated, faces = mesh.sample(2000, return_index=True)
    normals_rotated = mesh.face_normals[faces]
    xyzs_rotated = torch.as_tensor(xyzs_rotated).unsqueeze(0).to(next(model.parameters()))
    normals_rotated = torch.as_tensor(normals_rotated).unsqueeze(0).to(next(model.parameters()))
    # concatenate the xyzs and normals to get a 6D input
    feats_rotated = torch.cat([xyzs_rotated, normals_rotated], dim=2)
    # Pass the rotated points through the model
    if model.rotation_representation == "6d":
        up_predicted, front_predicted = model(feats_rotated)
        # Force front_predicted to be orthogonal to up_predicted
        front_predicted = front_predicted - torch.sum(front_predicted * up_predicted, dim=1, keepdim=True) * up_predicted
        # Normalize front_predicted again
        front_predicted = F.normalize(front_predicted, p=2, dim=1)
        # Compute the rotation matrix from the predictions
        predicted_rotations = rotation_from_model_outs(up_predicted, front_predicted).squeeze()
    elif model.rotation_representation == "procrustes":
        predicted_rotations = model(feats_rotated).squeeze()
    # Invert the predicted rotation matrix by taking the transpose
    inverse_predicted_rotation = (predicted_rotations.T).cpu().numpy()
    # Convert to 4x4 homogeneous matrix
    inverse_predicted_rotation = np.pad(inverse_predicted_rotation, ((0, 1), (0, 1)), mode='constant')
    inverse_predicted_rotation[-1, -1] = 1
    # Apply the predicted rotation to the mesh
    mesh.apply_transform(inverse_predicted_rotation)

    # also apply predicted_rotations to xyzs_rotated
    xyzs_rotated = xyzs_rotated @ predicted_rotations # don't transpose because we're right-multiplying
    # make a mesh from xyzs_rotated
    mesh_rotated = trimesh.Trimesh(vertices=xyzs_rotated.squeeze().cpu().numpy(), faces=[])
    # concatenate the two meshes
    # mesh = trimesh.util.concatenate([mesh, mesh_rotated])
    return mesh, mesh_rotated

def visualize_flipper_model_on_mesh(model, mesh, flip_matrices):
    """Visualize the effect of a flip on a mesh."""
    # Draw a random int from 0 to 23
    flip_idx = random.randint(0, 23)
    flip_matrix = flip_matrices[flip_idx] # (3, 3)
    # convert to 4x4 homogeneous matrix
    flip_matrix = np.pad(flip_matrix, ((0, 1), (0, 1)), mode='constant')
    flip_matrix[-1, -1] = 1
    # Apply the flip to the mesh
    mesh.apply_transform(flip_matrix)
    # Sample points from the mesh
    xyzs_flipped, faces = mesh.sample(2000, return_index=True)
    normals_flipped = mesh.face_normals[faces]
    xyzs_flipped = torch.as_tensor(xyzs_flipped).unsqueeze(0).to(next(model.parameters()))
    normals_flipped = torch.as_tensor(normals_flipped).unsqueeze(0).to(next(model.parameters()))
    # concatenate the xyzs and normals to get a 6D input
    feats_flipped = torch.cat([xyzs_flipped, normals_flipped], dim=2)
    # Pass the flipped points through the model
    logits = model(feats_flipped)
    predicted_flip_index = torch.argmax(logits, dim=1).squeeze()
    # Get the predicted flip matrix
    predicted_flip_matrix = flip_matrices[predicted_flip_index].T # want to invert this rotation
    # convert to 4x4 homogeneous matrix
    predicted_flip_matrix = np.pad(predicted_flip_matrix, ((0, 1), (0, 1)), mode='constant')
    predicted_flip_matrix[-1, -1] = 1
    # Apply the predicted flip to the mesh
    mesh.apply_transform(predicted_flip_matrix)
    return mesh