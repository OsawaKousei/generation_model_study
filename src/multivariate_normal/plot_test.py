import matplotlib.pyplot as plt
import numpy as np


def simple_3d_plot():
    x = np.array(
        [
            [-2, -1, 0, 1, 2],
            [-2, -1, 0, 1, 2],
            [-2, -1, 0, 1, 2],
            [-2, -1, 0, 1, 2],
            [-2, -1, 0, 1, 2],
        ]
    )

    y = np.array(
        [
            [-2, -2, -2, -2, -2],
            [-1, -1, -1, -1, -1],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
        ]
    )

    z = x**2 + y**2

    ax = plt.axes(projection="3d")
    ax.plot_surface(x, y, z, cmap="viridis")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


def mesh_grid_plot():
    xs = np.arange(-2, 2, 0.1)
    ys = np.arange(-2, 2, 0.1)
    x, y = np.meshgrid(xs, ys)
    z = x**2 + y**2

    ax = plt.axes(projection="3d")
    ax.plot_surface(x, y, z, cmap="viridis")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


def contour_plot():
    xs = np.arange(-2, 2, 0.1)
    ys = np.arange(-2, 2, 0.1)
    x, y = np.meshgrid(xs, ys)
    z = x**2 + y**2

    ax = plt.axes()
    ax.contour(x, y, z)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()


if __name__ == "__main__":
    # simple_3d_plot()
    # mesh_grid_plot()
    contour_plot()
