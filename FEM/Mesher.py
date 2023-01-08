import matplotlib.pyplot as plt
import numpy as np
import triangle as tr
import copy
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


def build_edges(vertices):
    edges = np.array([[i, i + 1] for i in range(len(vertices))])
    edges[-1, -1] = 0
    return edges


def tri3_2_tri6(mesh3):

    # translated function from matlab
    # takes a linear triangular element mesh and converts it to quadratic
    mesh6 = copy.copy(mesh3)  # we first copy the initial mesh

    sorted_nodes = mesh3.nodes[mesh3.connectivity.flatten(order='F')]  # we take every node's connectivity

    ne = mesh3.ne

    # this computes the in-between nodes to add to the linear triangle
    x1 = (sorted_nodes[:ne] + sorted_nodes[ne:2*ne])/2
    x2 = (sorted_nodes[ne:2*ne] + sorted_nodes[2*ne:])/2
    x3 = (sorted_nodes[:ne] + sorted_nodes[2*ne:])/2

    # we use the np.unique() function to map every point to it's original position
    new_nodes, ic = np.unique(np.vstack((x1, x2, x3)), return_inverse=True, axis=0)
    ic += mesh3.nn
    # we now add them to the mesh6 object
    mesh6.nodes = np.vstack((mesh3.nodes, new_nodes))
    mesh6.connectivity = np.hstack((mesh3.connectivity, np.vstack((ic[:ne], ic[ne:2*ne], ic[2*ne:])).T))
    mesh6.nn = len(mesh6.nodes)

    return mesh6


class Mesh:
    def __init__(self, vertices, edges, cell_size=5, simultype='2D'):

        A = dict(vertices=vertices, segments=edges, regions=np.array([[0.5, 0.5, 1, 0]]))
        out = tr.triangulate(A, f'pq30a{cell_size}')
        self._parse_mesh(out)
        self.simultype = simultype

    def _parse_mesh(self, out):

        self._mesh = out
        self.nodes = self._mesh['vertices']
        self.connectivity = self._mesh['triangles']

        self.ne = len(self.connectivity)
        self.nn = len(self.nodes)

        self.id = np.zeros(self.ne).astype(int)

    def refine(self, bbox, area=1):

        from scipy.spatial import Delaunay

        ch = Delaunay(bbox)
        nodes_in_hull = ch.find_simplex(self.nodes, tol=1e-3) > 0
        elem_in_hull = np.sum(np.isin(self.connectivity, np.argwhere(nodes_in_hull)[:, 0]), axis=1) > 0

        triangle_max_area = np.ones(len(elem_in_hull)) * area
        triangle_max_area[~elem_in_hull] = -1
        A = dict(vertices=self._mesh['vertices'],
                 vertex_markers=self._mesh['vertex_markers'],
                 triangles=self._mesh['triangles'],
                 triangle_max_area=triangle_max_area)

        out = tr.triangulate(A, f'rpq30a')
        self._parse_mesh(out)

    def fan_refine(self, center, radius, theta_min, theta_max, area=1, npoints=50):

        offset = (theta_max - theta_min) * 0.05

        theta = np.linspace(theta_min - offset, theta_max + offset, npoints)
        x = radius * np.hstack(([0], np.cos(theta))) + center[0]
        y = radius * np.hstack(([0], np.sin(theta))) + center[1]

        bbox = np.vstack((x, y)).T
        self.refine(bbox, area)

    def plot(self, z=None, c='k', shading='gouraud', ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()

        ax.triplot(*self.nodes.T, self.connectivity[:, :3], c=c, lw=0.5)
        if z is not None:
            im = ax.tripcolor(*self.nodes.T, self.connectivity[:, :3], z, shading=shading)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size=0.2, pad=0.1)
            cb = plt.colorbar(im, cax=cax)
            return fig, ax, cb

        return fig, ax
