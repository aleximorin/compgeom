import matplotlib.pyplot as plt
import numpy as np
import copy
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


def build_edges(vertices):
    edges = np.array([[i, i + 1] for i in range(len(vertices))]) % len(vertices)
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
    def __init__(self, meshio_mesh, cell_type='triangle', simultype='2D'):

        self.cell_type = cell_type
        self._parse_mesh(meshio_mesh)
        self.simultype = simultype

    def _parse_mesh(self, out):

        # we initially take the connectivity from the meshio object
        connectivity = out.cells_dict[self.cell_type].astype(int)

        # we need to filter the unused nodes that are residuals from pygmsh
        # did not find native way to do it
        used_nodes = np.zeros(len(out.points), dtype=bool)
        used_nodes[np.unique(connectivity)] = True
        nodes = out.points[used_nodes, :2]

        # we create an index map to reflect the new node indices
        index_map = np.zeros(len(out.points), dtype=int)
        index_map[used_nodes] = np.arange(len(nodes))
        new_connectivity = index_map[connectivity]

        # we now parse the arguments to te mesh object
        self.nodes = nodes
        self.connectivity = new_connectivity

        self.ne = len(self.connectivity)
        self.nn = len(self.nodes)

        self.id = np.zeros(self.ne).astype(int)

    def plot(self, z=None, c='k', shading='gouraud', ax=None, vmin=None, vmax=None, cmap='viridis'):
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect(1)
        else:
            fig = plt.gcf()

        ax.triplot(*self.nodes.T, self.connectivity[:, :3], c=c, lw=0.5)
        if z is not None:
            im = ax.tripcolor(*self.nodes.T, self.connectivity[:, :3], z, cmap=cmap,
                              shading=shading, vmin=vmin, vmax=vmax)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size=0.2, pad=0.1)
            cb = plt.colorbar(im, cax=cax)
            return fig, ax, cb

        return fig, ax


if __name__ == '__main__':

    # we will create a smiley face :)

    # we need to import pygmsh
    import pygmsh

    # cl is the mesh resolution
    cl = 0.1

    with pygmsh.geo.Geometry() as geom:
        # the face and the eyes need to be defined as 'not a surface' as we will create a proper surface later on
        # we create the face
        face = geom.add_circle([0.0, 0.0], 1, mesh_size=2 * cl, make_surface=False, compound=False)

        # we add two eyes
        eye1 = geom.add_circle([-0.4, 0.25], 0.2, mesh_size=cl, make_surface=False, compound=False)
        eye2 = geom.add_circle([0.4, 0.25], 0.2, mesh_size=cl, make_surface=False, compound=False)

        # we create a mouth by defining two circle arcs
        # left and right edge of the smile
        p1 = geom.add_point([0.7, -0.2], cl)
        p2 = geom.add_point([-0.7, -0.2], cl)

        # centers of the lower and upper part of the mouth
        c1 = geom.add_point([0, 2], cl)
        c2 = geom.add_point([0, -0.1], cl)

        # lower and upper part of the mouth
        m1 = geom.add_circle_arc(p1, c1, p2)
        m2 = geom.add_circle_arc(p2, c2, p1)

        # we create a loop by linking those two curves
        mouth = geom.add_curve_loop([m1, m2])

        # we can finally create the surface, not forgetting to add the holes
        surface = geom.add_plane_surface(face.curve_loop, holes=[eye1.curve_loop, eye2.curve_loop, mouth])

        # don't forget to generate the mesh, 'out' is what the wrapper needs
        out = geom.generate_mesh(dim=2)

    # we create the easily usable mesh object
    mesh = Mesh(out)

    # we can plot the mesh with the plot() command
    mesh.plot()

    # here we can also plot every nodes easily
    plt.scatter(*mesh.nodes.T, c='k', s=2)

    # obligatory plt.show() call for matplotlib to open a window
    plt.show()