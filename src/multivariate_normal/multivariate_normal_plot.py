import matplotlib.pyplot as plt
import numpy as np


def multivariate_normal(x, mu, cov):
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)

    D = len(x)
    z = 1 / (np.sqrt((2 * np.pi) ** D * det))
    y = z * np.exp((x - mu).T @ inv @ (x - mu) / -2.0)

    return y


if __name__ == "__main__":
    mu = np.array([0.5, -0.2])
    cov = np.array([[2.0, 0.3], [0.3, 0.5]])
    xs = ys = np.arange(-5, 5, 0.1)
    x, y = np.meshgrid(xs, ys)
    z = np.zeros_like(x)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            point = np.array([x[i, j], y[i, j]])
            z[i, j] = multivariate_normal(point, mu, cov)

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
