import numpy as np
import matplotlib.pyplot as plt
import Mesher
import Matrix

import PoroElasticProperties as prop

from scipy.sparse import linalg
from scipy.sparse import lil_matrix, csc_matrix

from scipy.io import loadmat

if __name__ == '__main__':
    import pygmsh

    # we define the mesh
    r = 1
    refine_radius = 2
    R = 30
    depth = r * R


    def tanh(dist, dist0, l0, l1, eps=1e-10):
        return l0 + l1 * np.tanh(dist / dist0) + eps


    with pygmsh.geo.Geometry() as geom:
        box = geom.add_polygon([[0, 0],
                                [R, 0],
                                [R, -depth],
                                [0, -depth]])

        geom.set_mesh_size_callback(lambda dim, tag, x, y, z, lc: tanh(np.sqrt((x - r) ** 2 + y ** 2), 30, 0.01, 3),
                                    ignore_other_mesh_sizes=False)

        out = geom.generate_mesh()

    mesh = Mesher.Mesh(out, simultype='axis')

    # we will try both a constant and linear strain triangle
    mesh = Mesher.tri3_2_tri6(mesh)

    # plt.xlim(-0.5, 5)
    # plt.ylim(-5, 0.5)

    matmesh = loadmat('../../mesh.mat')
    mesh.nodes = matmesh['nodes']
    mesh.connectivity = matmesh['connectivity'].astype(int) - 1
    mesh.ne = len(mesh.connectivity)
    mesh.nn = len(mesh.nodes)
    mesh.id = np.zeros(mesh.ne).astype(int)

    mesh.plot()

    # boundary conditions
    traction_boundary = np.argwhere((mesh.nodes[:, 0] <= r) & (mesh.nodes[:, 1] == 0))

    # applied load
    area = r * r * np.pi
    ts = [0, -100 * 1e3]
    fs = Matrix.assemble_tractions_over_line(mesh, traction_boundary, ts)

    # check it works, should output zero
    difference = 100 * (fs.sum() - ts[1] * area)/(ts[1] * area)
    print(f'relative difference is {difference:.2f} %')
    # lets gooo

    # outer boundaries
    bottom = np.argwhere((mesh.nodes[:, 1] == -depth))
    right = np.argwhere((mesh.nodes[:, 0] == R))
    left = np.argwhere((mesh.nodes[:, 0] == 0))
    top = np.argwhere((mesh.nodes[:, 1] == 0))

    fixed_nodes = np.unique(np.vstack((bottom, right)))
    fixed_dof = np.unique(np.hstack((fixed_nodes * 2, fixed_nodes * 2 + 1)))

    E = 20 * 1e6
    nu = 0.3

    K = Matrix.assemble_stiffness_matrix(mesh, E, nu)

    # solving the system
    u_set = fixed_dof * 0
    eq_to_solve = np.setdiff1d(np.arange(mesh.nn * 2), fixed_dof)
    f = -K[eq_to_solve][:, fixed_dof].dot(u_set) + fs[eq_to_solve]

    u_aux = linalg.spsolve(K[eq_to_solve][:, eq_to_solve], f)
    u_res = np.zeros(mesh.nn * 2)
    u_res[eq_to_solve] = u_aux

    udisp = u_res.reshape(2, -1, order='F')

    Matrix.project_stress(mesh, E, nu, u_res)
