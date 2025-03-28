import numpy as np
import matplotlib.pyplot as plt
import poisson_function_define as pof


x1, x2 = 0, 1
y1, y2 = 0, 1
m, n = 100, 100

x = np.linspace(x1, x2, m + 1)
y = np.linspace(y1, y2, n + 1)
X, Y = np.meshgrid(x, y)

U = pof.exact_a(X, Y)

# 可视化
plt.imshow(U, extent=[x1, x2, y1, y2], cmap="jet", origin="lower")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.show()
