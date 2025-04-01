import numpy as np
from FDM_piosson_optimization import *
from scipy.sparse import csr_matrix
from tqdm import tqdm
import poisson_function_define as pod


def generate_poisson_dataset(
    num_samples=1000, hx=0.01, hy=0.01, save_path="poisson_dataset.npz"
):
    x = np.array([0, 1])
    y = np.array([0, 1])
    pde = PDE2DModel_OPT(x, y)

    f_data = []
    u_data = []

    print(f"Generating {num_samples} samples...")
    for _ in tqdm(range(num_samples)):
        X, Y, U = NDM5_2D(pde, hx, hy)
        F = pod.f_guassian_kernel_combine_2(X, Y)
        f_data.append(F.astype(np.float32))
        u_data.append(U.astype(np.float32))

    f_data = np.stack(f_data)  # Shape: (num_samples, H, W)
    u_data = np.stack(u_data)

    np.savez_compressed(save_path, f=f_data, u=u_data)
    print(f"Dataset saved to {save_path} ✅")


if __name__ == "__main__":
    generate_poisson_dataset(
        num_samples=1000, hx=0.01, hy=0.01, save_path="poisson_dataset_1000.npz"
    )
