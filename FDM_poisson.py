import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as ss
from scipy.sparse.linalg import spsolve
import poisson_function_define as pod


# most precise
class PDE2DModel:

    def __init__(self, x, y):
        assert len(x) == len(y) == 2, "ERROR:UNEXPECTED SV and EV!!"
        self.x = x
        self.y = y

    def space_grid(self, hx, hy):
        M = int(round((self.x[1] - self.x[0]) / hx, 0))
        N = int(round((self.y[1] - self.y[0]) / hy, 0))
        assert M == N >= 3, "mesh number is right"
        X = np.linspace(self.x[0], self.x[1], M + 1)
        Y = np.linspace(self.y[0], self.y[1], N + 1)
        return M, N, X, Y


def NDM5_2D(PDE2DModel, hx, hy):
    M, N, X0, Y0 = PDE2DModel.space_grid(hx, hy)
    Y, X = np.meshgrid(Y0, X0)
    U = np.zeros((M + 1, N + 1))
    U[0, :] = pod.left_guassian_kernel(Y0)
    U[-1, :] = pod.right_guassian_kernel(Y0)
    U[:, 0] = pod.down_guassian_kernel(X0)
    U[:, -1] = pod.up_guassian_kernel(X0)
    # define some number
    p = 1 / hx**2
    q = 1 / hy**2
    r = -2 * (p + q)
    # define matrix
    D = np.diag([q for i in range(M - 1)])
    C = np.zeros((N - 1, N - 1), dtype="float64")
    for i in range(N - 1):
        C[i][i] = r
        if i < N - 2:
            C[i][i + 1] = p
            C[i + 1][i] = p

    u0 = np.array([[pod.down_guassian_kernel(X0[i])] for i in range(1, M)])
    un = np.array([[pod.up_guassian_kernel(X0[i])] for i in range(1, M)])
    F = np.zeros((M - 1) * (N - 1))
    for j in range(1, N):
        for i in range(1, M):
            F[(N - 1) * (j - 1) : (N - 1) * (j)] = pod.f_guassian_kernel(
                X0[1:M], np.array([Y0[j] for i in range(N - 1)]), 2, 2, 1
            )

            F[(N - 1) * (j - 1)] -= pod.left_guassian_kernel(Y0[j]) * p
            F[(N - 1) * (j) - 1] -= pod.right_guassian_kernel(Y0[j]) * p
    F[: N - 1] -= np.dot(D, u0).T[0]
    F[(M - 1) * (N - 2) :] -= np.dot(D, un).T[0]
    F = np.asmatrix(F).T

    Dnew = np.zeros(((M - 1) * (N - 1), (N - 1) * (M - 1)))
    for i in range((N - 1) * (N - 1)):
        Dnew[i][i] = r
        if i < (N - 2) * (N - 1):
            Dnew[i][i + N - 1] = q
            Dnew[i + N - 1][i] = q
    for i in range(N - 1):
        for j in range(N - 2):
            Dnew[(N - 1) * i + j][(N - 1) * i + j + 1] = p
            Dnew[(N - 1) * i + j + 1][(N - 1) * i + j] = p
    print("FDM define finish, start solving...")
    Unew = np.linalg.solve(Dnew, F)
    U[1:-1, 1:-1] = Unew[:, 0].reshape((N - 1, N - 1)).T
    return X, Y, U


x = np.array([0, 1])
y = np.array([0, 1])

pde = PDE2DModel(x, y)
X, Y, U = NDM5_2D(pde, 0.01, 0.01)
plt.pcolormesh(X, Y, U, cmap="jet", shading="auto")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("u_f")
plt.show()
