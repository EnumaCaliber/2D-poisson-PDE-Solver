import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# Use English for all text
plt.rcParams["font.family"] = "DejaVu Sans"

# Parameter settings - uniform 1x1 domain and grid
Nx = 100  # Grid points in x direction
Ny = 100  # Grid points in y direction (same as x)
Nt = 1000

D = 0.005
x_start = 0
x_end = 1  # Domain from 0 to 1
y_start = 0
y_end = 1  # Domain from 0 to 1
t_start = 0
t_end = 2.5

xx = np.linspace(x_start, x_end, Nx + 1, endpoint=True)
yy = np.linspace(y_start, y_end, Ny + 1, endpoint=True)
tt = np.linspace(t_start, t_end, Nt + 1, endpoint=True)

dx = (x_end - x_start) / Nx  # dx = 1/100 = 0.01
dy = (y_end - y_start) / Ny  # dy = 1/100 = 0.01 (same as dx)
dt = (t_end - t_start) / Nt


def set_vec_field():
    xx_mesh, yy_mesh = np.meshgrid(xx, yy)
    v = np.zeros_like(xx_mesh)
    u = 4 * (1 - yy_mesh) * yy_mesh  # Parabolic velocity distribution
    return xx_mesh, yy_mesh, u, v


def gaussian_blob(x_mesh, y_mesh, x_center, y_center, amplitude=1.0, width=0.01):
    """Create a Gaussian blob at specified location"""
    return amplitude * np.exp(
        -((x_mesh - x_center) ** 2 + (y_mesh - y_center) ** 2) / width
    )


def random_gaussian_superposition(xx_mesh, yy_mesh, num_blobs, seed=None):
    """Create random Gaussian superposition initial condition"""

    if seed is not None:
        np.random.seed(seed)

    # Initialize result
    result = np.zeros_like(xx_mesh)

    # Generate random blobs
    for i in range(num_blobs):
        # Random center position (with margin from boundaries)
        x_center = np.random.uniform(0.1, 0.9)
        y_center = np.random.uniform(0.1, 0.9)

        # Random amplitude (between 0.3 and 1.5)
        amplitude = np.random.uniform(0.3, 1.5)

        # Random width (between 0.002 and 0.02)
        width = np.random.uniform(0.002, 0.02)

        # Add Gaussian blob
        blob = gaussian_blob(xx_mesh, yy_mesh, x_center, y_center, amplitude, width)
        result += blob

    return result


# User input for random seed
seed = random.randint(0, 2000)
num_blobs = random.randint(3, 10)


xx_mesh, yy_mesh, u, v = set_vec_field()

# Calculate coefficients
Ax = u[1:-1, 1:-1] * dt / dx
Ay = v[1:-1, 1:-1] * dt / dx
Bx = D * dt / dx / dx
By = D * dt / dy / dy

# Initialize temperature field with random Gaussian superposition
T_table = np.zeros((Nt + 1, Ny + 1, Nx + 1))
T_table[0, :, :] = random_gaussian_superposition(xx_mesh, yy_mesh, num_blobs, seed)

# First step calculation
i = 0
T_table[i + 1, 1:-1, 1:-1] = (
    (0.5 * (Ax - 1) * Ax + Bx) * T_table[i, 1:-1, 2:]
    + (0.5 * (Ay - 1) * Ay + By) * T_table[i, 2:, 1:-1]
    + (1 - Ax * Ax - Ay * Ay - 2 * Bx - 2 * By) * T_table[i, 1:-1, 1:-1]
    + (0.5 * (1 + Ax) * Ax + Bx) * T_table[i, 1:-1, :-2]
    + (0.5 * (1 + Ay) * Ay + By) * T_table[i, :-2, 1:-1]
)

# DFF scheme solution
print("Computing...")
for i in range(1, len(T_table) - 1):
    if i % 100 == 0:
        print(f"Progress: {i}/{Nt}")

    T_table[i + 1, 1:-1, 1:-1] = (
        (1 - 2 * Bx - 2 * By) / (1 + 2 * Bx + 2 * By) * T_table[i - 1, 1:-1, 1:-1]
        + (-Ax + 2 * Bx) / (1 + 2 * Bx + 2 * By) * T_table[i, 1:-1, 2:]
        + (Ax + 2 * Bx) / (1 + 2 * Bx + 2 * By) * T_table[i, 1:-1, :-2]
        + (-Ay + 2 * By) / (1 + 2 * Bx + 2 * By) * T_table[i, 2:, 1:-1]
        + (Ay + 2 * By) / (1 + 2 * Bx + 2 * By) * T_table[i, :-2, 1:-1]
    )

print("Starting animation...")

# Create animation
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
levels = np.linspace(0, np.max(T_table), 51, endpoint=True)


def update_animation(frame):
    time_step = frame * 3
    if time_step >= Nt:
        time_step = Nt

    ax.clear()

    contour = ax.contourf(
        xx_mesh,
        yy_mesh,
        T_table[time_step, :, :],
        levels=levels,
        cmap="hot",
        extend="max",
    )

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return (contour,)


# Create animation
animation_frames = Nt // 3
anim = FuncAnimation(
    fig, update_animation, frames=animation_frames, interval=30, blit=False, repeat=True
)

# Add color bar
plt.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
temp_contour = ax.contourf(
    xx_mesh, yy_mesh, T_table[0, :, :], levels=levels, cmap="hot"
)
cbar = fig.colorbar(temp_contour, cax=cbar_ax)
cbar.set_label("Temperature", fontsize=12)

plt.tight_layout()
plt.show()
