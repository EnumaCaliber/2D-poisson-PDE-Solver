import numpy as np


# define function a
def exact_a(x, y):
    return np.exp(x * y)


def f_a(x, y):
    return (np.exp(x * y)) * (x**2 + y**2)


def left_a(y):
    return np.exp(0 * y)


def right_a(y):
    return np.exp(1 * y)


def down_a(x):
    return np.exp(0 * x)


def up_a(x):
    return np.exp(1 * x)


# define function b
k = 6


def f_b(x, y):
    return 1


def left_b(y):
    return np.exp(k * 0) * np.sin(k * y) + 0.25 * (0**2 + y**2)


def right_b(y):
    return np.exp(k * 1) * np.sin(k * y) + 0.25 * (1**2 + y**2)


def down_b(x):
    return np.exp(k * x) * np.sin(k * 0) + 0.25 * (x**2 + 0**2)


def up_b(x):
    return np.exp(k * x) * np.sin(k * 1) + 0.25 * (x**2 + 1**2)


def exact_b(x, y):
    return np.exp(k * x) * np.sin(k * y) + 0.25 * (x**2 + y**2)


# define function c
def exact_c(x, y):
    return np.sinh(x)


def f_c(x, y):
    return np.sinh(x)


def left_c(y):
    return np.sinh(0) + 0 * y


def right_c(y):
    return np.sinh(1) + 0 * y


def down_c(x):
    return np.sinh(x)


def up_c(x):
    return np.sinh(x)


# define function d
def f_d(x, y):
    return 4 * (x**2 + y**2 + 1) * np.exp(x**2 + y**2)


def exact_d(x, y):
    return np.exp(x**2 + y**2)


def left_d(y):
    return np.exp(0**2 + y**2)


def right_d(y):
    return np.exp(1**2 + y**2)


def down_d(x):
    return np.exp(x**2 + 0**2)


def up_d(x):
    return np.exp(x**2 + 1**2)


# define function e
def f_e(x, y):
    return np.exp(x * y) * (x**2 + y**2) + np.sinh(x)


def exact_e(x, y):
    return np.exp(x * y) + np.sinh(x)


def left_e(y):
    return np.exp(0 * y) + np.sinh(0)


def right_e(y):
    return np.exp(1 * y) + np.sinh(1)


def down_e(x):
    return np.exp(x * 0) + np.sinh(x)


def up_e(x):
    return np.exp(x * 1) + np.sinh(x)


# define guassian kernel
def f_guassian_kernel(x, y, c1, c2, sigma):
    return 2 * np.exp(((x - c1) ** 2 + (y - c2) ** 2) / sigma**2)


def left_guassian_kernel(y):
    return 0 * y


def right_guassian_kernel(y):
    return 0 * y


def down_guassian_kernel(x):
    return 0 * x


def up_guassian_kernel(x):
    return 0 * x


# define guassian kernel combine
def f_guassian_kernel_combine(x, y, c1, c2, sigma):
    return 3 * np.exp(((x - c1) ** 2 + (y - c2) ** 2) / sigma**2) - 1 * np.exp(
        ((x - 1) ** 2 + (y - 2) ** 2) / 1**2
    )


def left_guassian_kernel_combine(y):
    return 0 * y


def right_guassian_kernel_combine(y):
    return 0 * y


def down_guassian_kernel_combine(x):
    return 0 * x


def up_guassian_kernel_combine(x):
    return 0 * x


def f_guassian_kernel_diffdis(x, y, c1, c2, sigma1, sigma2, b):
    return b * np.exp(
        -((x - c1) ** 2) / (2 * sigma1**2) - (y - c2) ** 2 / (2 * sigma2**2)
    )
