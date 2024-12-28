import matplotlib.pyplot as plt
import numpy as np

mus = np.array([[2.0, 54.50], [4.3, 80.0]])
covs = np.array([[[0.07, 0.44], [0.44, 33.7]], [[0.17, 0.94], [0.94, 36.00]]])
phis = np.array([0.35, 0.65])


def multivariate_normal(x, mu, cov):
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)

    D = len(x)
    z = 1 / (np.sqrt((2 * np.pi) ** D * det))
    y = z * np.exp((x - mu).T @ inv @ (x - mu) / -2.0)

    return y


def gmm(x, phis, mus, covs):
    k = len(phis)
    y = 0
    for i in range(k):
        phi, mu, cov = phis[i], mus[i], covs[i]
        y += phi * multivariate_normal(x, mu, cov)

    return y


xs = np.arange(1, 6, 0.1)
ys = np.arange(40, 100, 0.1)
x, y = np.meshgrid(xs, ys)
z = np.zeros_like(x)

for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        point = np.array([x[i, j], y[i, j]])
        z[i, j] = gmm(point, phis, mus, covs)

fig = plt.figure()
ax1 = fig.add_subplot(121, projection="3d")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
ax1.plot_surface(x, y, z, cmap="viridis")

ax2 = fig.add_subplot(122)
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.contour(x, y, z)

plt.show()
