import os, random
import trimesh, torch
from torch.utils.data import Dataset, DataLoader
from utils.helpers import small_angle_random_rotation_matrix
import numpy as np
from tqdm import tqdm

class MultiMeshFlipperDataset(Dataset):
    def __init__(self, index_file_path, sample_size, preload=False, confusion_matrices=False):
        # sample_size is number of points sampled per fetch
        self.sample_size = sample_size
        self.preload = preload
        self.confusion_matrices = confusion_matrices

        # Load the index file
        with open(index_file_path, "r") as f:
            self.index_list = f.readlines() # this is a list of paths to meshes
            self.index_list = [fname.strip() for fname in self.index_list]

        # set flag based on extension of first file
        ext = os.path.splitext(self.index_list[0])[1]
        if ext == ".obj" or ext == ".off":
            self.mesh_or_point_cloud = "mesh"
        elif ext == ".npy":
            self.mesh_or_point_cloud = "point_cloud"
        else:
            raise ValueError("Unknown file extension")

        self.dataset_size = len(self.index_list)
        print(f"Dataset size: {self.dataset_size}")

        if self.preload:
            print("Preloading meshes into memory")
            if self.mesh_or_point_cloud == "mesh":
                # load all meshes into memory
                self.xyzs = []
                self.normals = []
                for meshpath in tqdm(self.index_list):
                    tmesh = trimesh.load(meshpath, force="mesh")
                    verts, faces = self.normalize_mesh(tmesh.vertices, tmesh.faces)
                    tmesh = trimesh.Trimesh(verts, faces)
                    xyz, faces = tmesh.sample(self.sample_size, return_index=True)
                    xyz = torch.as_tensor(xyz, dtype=torch.float32)
                    normals = torch.as_tensor(tmesh.face_normals[faces], dtype=torch.float32) # face normals at sampled points
                    self.xyzs.append(xyz)
                    self.normals.append(normals)
            elif self.mesh_or_point_cloud == "point_cloud":
                # load all point clouds and normals into memory
                self.xyzs = []
                self.normals = []
                if self.confusion_matrices:
                    self.confusion_matrices = []
                for fname in tqdm(self.index_list):
                    xyz = torch.tensor(np.load(fname), dtype=torch.float32)
                    # path for normals is same as xyz with "point_cloud" replaced by "normals"
                    normals = torch.tensor(np.load(fname.replace("point_cloud", "normals")), dtype=torch.float32)
                    self.xyzs.append(xyz)
                    self.normals.append(normals)
                    if self.confusion_matrices:
                        # path for confusion matrix is same as xyz with "point_cloud" replaced by "confusion_matrix"
                        confusion_matrix = torch.tensor(np.load(fname.replace("point_cloud", "confusion_mtx")), dtype=torch.float32)
                        self.confusion_matrices.append(confusion_matrix)

        # dataset comes with a fixed dict of 24 rotation matrices corresponding to 24 cube symmetries
        self.cube_flips = torch.load("utils/24_cube_flips.pt") # (24, 3, 3)

        self.s = 10 * np.pi / 180 # 10 degrees in radians
        print("Max rotation noise angle in degrees: ", self.s * 180 / np.pi)


    def __getitem__(self, idx):
        # first load mesh
        if self.preload:
            if self.mesh_or_point_cloud == "mesh":
                idx = idx % len(self.xyzs)
                xyz = self.xyzs[idx]
                normals = self.normals[idx]
            elif self.mesh_or_point_cloud == "point_cloud":
                xyz = self.xyzs[idx]
                normals = self.normals[idx]
                if self.confusion_matrices:
                    confusion_matrix = self.confusion_matrices[idx]
                # subsample self.sample_size points
                if xyz.shape[0] > self.sample_size:
                    subsampled_indices = torch.randperm(xyz.shape[0])[:self.sample_size]
                    xyz = xyz[subsampled_indices]
                    normals = normals[subsampled_indices]

        else:
            if self.mesh_or_point_cloud == "mesh":
                meshpath = self.index_list[idx]
                tmesh = trimesh.load(meshpath, force="mesh") # force to load as mesh
                verts, faces = self.normalize_mesh(tmesh.vertices, tmesh.faces)
                tmesh = trimesh.Trimesh(verts, faces)
                xyz, faces = tmesh.sample(self.sample_size, return_index=True)
                xyz = torch.as_tensor(xyz, dtype=torch.float32)
                normals = torch.as_tensor(tmesh.face_normals[faces], dtype=torch.float32) # face normals at sampled points
            elif self.mesh_or_point_cloud == "point_cloud":
                xyz = torch.tensor(np.load(self.index_list[idx]), dtype=torch.float32)
                # path for normals is same as xyz with "point_cloud" replaced by "normals"
                normals = torch.tensor(np.load(self.index_list[idx].replace("point_cloud", "normals")), dtype=torch.float32)
                if self.confusion_matrices:
                    # path for confusion matrix is same as xyz with "point_cloud" replaced by "confusion_matrix"
                    confusion_matrix = torch.tensor(np.load(self.index_list[idx].replace("point_cloud", "confusion_mtx")), dtype=torch.float32)
                # subsample self.sample_size points
                if xyz.shape[0] > self.sample_size:
                    subsampled_indices = torch.randperm(xyz.shape[0])[:self.sample_size]
                    xyz = xyz[subsampled_indices]
                    normals = normals[subsampled_indices]

        # Generate a small random rotation to model error from the first-stage outputs
        rotation_noise_matrix = small_angle_random_rotation_matrix(self.s).to(xyz) # (3, 3)
        noisy_xyz = xyz @ rotation_noise_matrix.t() # (N, 3)

        # Also rotate the normals
        noisy_normals = normals @ rotation_noise_matrix.t()

        # Draw a random int from 0 to 23
        flip_idx = random.randint(0, 23)
        flip_matrix = self.cube_flips[flip_idx].to(xyz) # (3, 3)

        # Apply the flip to the noisy_xyz
        xyz_flipped = noisy_xyz @ flip_matrix.t() # (N, 3)

        # Apply the flip to the noisy_normals
        normals_flipped = noisy_normals @ flip_matrix.t()

        if self.confusion_matrices:
            return idx, xyz_flipped, normals_flipped, flip_idx, rotation_noise_matrix, confusion_matrix
        else:
            return idx, xyz_flipped, normals_flipped, flip_idx, rotation_noise_matrix
    
    def __len__(self):
        return self.dataset_size
    
    @staticmethod
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

def main():
    meshdir = "meshes"
    dataset = MultiMeshFlipperDataset(meshdir, sample_size = 100, dataset_size = 100)
    dataloader = DataLoader(dataset, batch_size = 5)
    for i, batch in enumerate(dataloader):
        indices, xyzs_flipped, normals_flipped, flip_indices, rotation_noise_matrices = batch
        print(i, indices)

        pass

    pass

if __name__ == '__main__':
    main()