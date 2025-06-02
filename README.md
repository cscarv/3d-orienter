Welcome to the code for [Symmetry-Robust 3D Orientation Estimation](https://arxiv.org/abs/2410.02101), which has been accepted for publication at ICML 2025.

### Some pointers:

- We use `poetry` for package and dependency management. Please follow the instructions in the [docs](https://python-poetry.org/docs/) to install and use `poetry`. Our method's dependencies are stored in `pyproject.toml`, and poetry will install all the necessary packages when you execute `poetry install`.

- Most of our training and evaluation scripts require the user to pass index files as arguments. 
    - These should be `.txt` files consisting of N lines, each storing the path to either (1) a `.npy` file containing point clouds or normal vectors sampled from a single shape (used in training and validation), or (2) a `.obj` file storing a mesh for a single shape (used at inference time). 
    - We've provided a simple script for generating an index file from a directory of meshes in `utils/generate_index_file.py`. 

- We have included a script `utils/presample_point_clouds_from_mesh.py` to help users normalize their meshes and presample point clouds and surface normals for training. Please normalize your meshes before training a model or performing inference using our pretrained checkpoints.

- We have included pretrained checkpoints for our orienter and flipper for your convenience in the folder `pretrained_ckpts`.

- You can use `two_stage_inference_script.py` to orient meshes in `.obj` format with our pipeline. Pass an `inference_index_file` storing paths to the meshes to orient as an argument, and edit `results_subdir` to control where the oriented meshes are saved.

### Citing our work

If you use this codebase in your research, please cite our paper: 

```
@inproceedings{
symmetry-robust-orientation2025, 
title = {Symmetry-Robust 3D Orientation Estimation}, 
author = {Christopher Scarvelis and David Benhaim and Paul Zhang}, 
booktitle = {Forty-second International Conference on Machine Learning}, 
year = {2025}, 
url = {https://openreview.net/forum?id=rcDYnkG1F8} 
}