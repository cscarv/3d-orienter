import os
import trimesh, torch
from torch.utils.data import Dataset, DataLoader
from utils.helpers import random_rotation_matrix
import numpy as np
from tqdm import tqdm

class MultiMeshDataset(Dataset):
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
                        # path for confusion matrix is same as xyz with "point_cloud" replaced by "confusion_mtx"
                        confusion_mtx = torch.tensor(np.load(fname.replace("point_cloud", "confusion_mtx")), dtype=torch.float32)
                        self.confusion_matrices.append(confusion_mtx)


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
                    confusion_mtx = self.confusion_matrices[idx]
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
                    # path for confusion matrix is same as xyz with "point_cloud" replaced by "confusion_mtx"
                    confusion_mtx = torch.tensor(np.load(self.index_list[idx].replace("point_cloud", "confusion_mtx")), dtype=torch.float32)
                # subsample self.sample_size points
                if xyz.shape[0] > self.sample_size:
                    subsampled_indices = torch.randperm(xyz.shape[0])[:self.sample_size]
                    xyz = xyz[subsampled_indices]
                    normals = normals[subsampled_indices]

        # Generate random rotation matrix and rotate the points
        target_rotation_matrix = random_rotation_matrix().to(xyz) # casts to same device and dtype as xyz
        xyz_rotated = xyz @ target_rotation_matrix.t()

        # Also rotate the normals
        normals_rotated = normals @ target_rotation_matrix.t()

        # Return rotated points and the target rotation matrix
        # Need to multiply xyz_rotated by transpose of target_rotation_matrix to get back the original points

        if self.confusion_matrices:
            return idx, xyz_rotated, target_rotation_matrix, normals_rotated, confusion_mtx
        else:
            return idx, xyz_rotated, target_rotation_matrix, normals_rotated
    
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
    dataset = MultiMeshDataset(meshdir, sample_size = 100, dataset_size = 100)
    dataloader = DataLoader(dataset, batch_size = 5)
    for i, batch in enumerate(dataloader):
        indices, xyzs_rotated, target_rotation_matrices, normals = batch
        print(i, indices)
        pass
    pass

if __name__ == '__main__':
    main()