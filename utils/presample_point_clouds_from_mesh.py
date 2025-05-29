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

def presample_point_clouds_from_mesh(index_file_path, num_points=10000):
    # Load the index file
    with open(index_file_path, "r") as f:
        index_list = f.readlines() # this is a list of paths to meshes
        index_list = [fname.strip() for fname in index_list]
        
    for meshpath in tqdm(index_list):
        tmesh = trimesh.load(meshpath, force="mesh")
        verts, faces = normalize_mesh(tmesh.vertices, tmesh.faces)
        tmesh = trimesh.Trimesh(verts, faces)
        xyz, faces = tmesh.sample(num_points, return_index=True)
        normals = tmesh.face_normals[faces]
        # convert to contiguous format
        xyz = np.array(xyz, order="C")
        normals = np.array(normals, order="C")
        # Save the point cloud to a file -- same directory as the mesh
        xyz_outpath = meshpath.replace("model_normalized.obj", "point_cloud.npy")
        np.save(xyz_outpath, xyz)
        normals_outpath = meshpath.replace("model_normalized.obj", "normals.npy")
        np.save(normals_outpath, normals)

if __name__ == "__main__":
    index_file_path = "data/shapenet_index_files/all_meshes/all.txt"
    presample_point_clouds_from_mesh(index_file_path)