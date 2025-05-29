import torch

def main():
    # Generate 6 3x3 permutation matrices
    perm_list = [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]
    permutation_matrices = torch.eye(3).repeat(6,1,1)
    for i, perm in enumerate(perm_list):
        permutation_matrices[i] = permutation_matrices[i][:,perm]

    # Generate every possible sign permutation of the columns of the permutation matrices
    sign_permutations = torch.tensor([[1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1], [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]])

    rotation_matrices = torch.zeros(48,3,3)
    for i in range(6):
        for j in range(8):
            rotation_matrices[i*8+j] = permutation_matrices[i]*sign_permutations[j]

    # keep only the rotation matrices with positive determinant
    determinants = torch.linalg.det(rotation_matrices)
    rotation_matrices = rotation_matrices[determinants == 1]
    
    # Save the tensor
    torch.save(rotation_matrices, "24_cube_flips.pt")

if __name__ == '__main__':
    main()