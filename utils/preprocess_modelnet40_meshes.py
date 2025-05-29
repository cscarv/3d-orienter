import trimesh
import numpy as np
from tqdm import tqdm

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

def preprocess_modelnet_meshes(index_list, num_points=10000):
    for meshpath in tqdm(index_list):
        tmesh = trimesh.load(meshpath, force="mesh")
        verts, faces = normalize_mesh(tmesh.vertices, tmesh.faces)
        tmesh = trimesh.Trimesh(verts, faces)
        # apply rotations to bring meshes into upright orientation -- CANNOT automate consistent front-facing orientations
        rotation = trimesh.transformations.rotation_matrix(-np.pi/2, (1, 0, 0))
        tmesh.apply_transform(rotation)
        xyz, faces = tmesh.sample(num_points, return_index=True)
        normals = tmesh.face_normals[faces]
        # convert to contiguous format
        xyz = np.array(xyz, order="C")
        normals = np.array(normals, order="C")
        # Save the point cloud to a file -- same directory as the mesh
        xyz_outpath = meshpath.replace(".off", "_point_cloud.npy")
        np.save(xyz_outpath, xyz)
        normals_outpath = meshpath.replace(".off", "_normals.npy")
        np.save(normals_outpath, normals)
        # also save the mesh
        tmesh.export(meshpath.replace(".off", "_normalized_rotated.off"))

if __name__ == "__main__":
    train_index_file_path = "data/modelnet40_index_files/train.txt"
    val_index_file_path = "data/modelnet40_index_files/val.txt"

    # load the train index file
    with open(train_index_file_path, "r") as f:
        train_index_list = f.readlines() # this is a list of paths to meshes
        train_index_list = [fname.strip() for fname in train_index_list]

    # load the val index file
    with open(val_index_file_path, "r") as f:
        val_index_list = f.readlines() # this is a list of paths to meshes
        val_index_list = [fname.strip() for fname in val_index_list]
    
    index_list = train_index_list + val_index_list
    preprocess_modelnet_meshes(index_list)