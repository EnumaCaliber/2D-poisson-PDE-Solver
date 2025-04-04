import numpy as np
from FDM_piosson_optimization import *
from scipy.sparse import csr_matrix
from tqdm import tqdm
import poisson_function_define as pod


def generate_poisson_dataset(
    num_samples=200, hx=0.01, hy=0.01, save_path="poisson_dataset.npz"
):
    x = np.array([0, 1])
    y = np.array([0, 1])
    pde = PDE2DModel_OPT(x, y)

    u_data = []

    print(f"Generating {num_samples} samples...")
    for _ in tqdm(range(num_samples)):
        X, Y, U = NDM5_2D(pde, hx, hy)
        u_data.append(U.astype(np.float32))
    # Shape: (num_samples, H, W)
    u_data = np.stack(u_data)

    np.savez_compressed(save_path, image=u_data)
    print(f"Dataset saved to {save_path} ✅")


if __name__ == "__main__":
    generate_poisson_dataset(
        num_samples=200, hx=0.01, hy=0.01, save_path="poisson_dataset_200.npz"
    )
