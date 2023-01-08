import numpy as np
import matplotlib.pyplot as plt
from Mesher import Mesh
import Matrix
from scipy.sparse import linalg

if __name__ == '__main__':

    h1 = 10
    h2 = 7
    l1 = 5
    l2 = 5
    lwall = 0.01
    hwall = 5

    vertices = np.array([[0, 0],
                         [0, h1],
                         [l1 - lwall, h1],
                         [l1 - lwall, h1 - hwall],
                         [l1 + lwall, h1 - hwall],
                         [l1 + lwall, h2],
                         [l1 + lwall + l2, h2],
                         [l1 + lwall + l2, 0]])

    edges = np.array([[i, i + 1] for i in range(len(vertices))])
    edges[-1, -1] = 0

    mesh = Mesh(vertices, edges, cell_size=5)

    left = np.argwhere(mesh.nodes[:, 0] == 0)
    right = np.argwhere(mesh.nodes[:, 0] == mesh.nodes[:, 0].max())

    bottom = np.argwhere(mesh.nodes[:, 1] == 0)
    top_left = np.argwhere((mesh.nodes[:, 1] == h1) & (mesh.nodes[:, 0] <= l1))
    top_right = np.argwhere((mesh.nodes[:, 1] == h2) & (mesh.nodes[:, 0] >= l1 + lwall))

    mesh.plot()
    for cond in [left, right, bottom, top_left, top_right]:
        plt.scatter(mesh.nodes[cond, 0], mesh.nodes[cond, 1])

    # enforcing boundary conditions
    nodes_fixed = np.unique(np.vstack((top_left, top_right)))
    h_fixed = mesh.nodes[nodes_fixed, 1]
    nodes_unknown = np.setdiff1d(np.arange(mesh.nn), nodes_fixed)

    C = Matrix.assemble_conductivity_matrix(mesh, 1)

    C_fixed = C[nodes_unknown][:, nodes_fixed]
    C_unknown = C[nodes_unknown][:, nodes_unknown]

    A = -C_fixed @ h_fixed
    h_unknown = linalg.inv(C_unknown) @ A

    h = np.zeros(mesh.nn)
    h[nodes_unknown] = h_unknown
    h[nodes_fixed] = h_fixed

    mesh.plot(h)
    plt.show()