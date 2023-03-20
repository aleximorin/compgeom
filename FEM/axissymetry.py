import numpy as np
import matplotlib.pyplot as plt
import Mesher
import Matrix

import PoroElasticProperties as prop

from scipy.sparse import linalg
from scipy.sparse import lil_matrix, csc_matrix

from scipy.io import loadmat

if __name__ == '__main__':
    # we define the mesh
    r = 1
    R = 30
    depth = r * R

    nodes = np.array([[0, 0],
                      [r, 0],
                      [R, 0],
                      [R, -depth],
                      [0, -depth]])

    # this will be sufficient for now but we need a better mesher
    edges = Mesher.build_edges(nodes)
    mesh = Mesher.Mesh(nodes, edges, cell_size=1, simultype='axis')
    matmesh = loadmat('../../mesh.mat')
    mesh.nodes = matmesh['nodes']
    mesh.connectivity = matmesh['connectivity'] - 1
    mesh.ne = len(mesh.connectivity)
    mesh.nn = len(mesh.nodes)
    mesh.id = np.ones(mesh.nn)

    mesh.plot()

    # we will try both a constant and linear strain triangle
    # mesh = Mesher.tri3_2_tri6(mesh)

    # boundary conditions
    traction_boundary = np.argwhere((mesh.nodes[:, 0] <= r) & (mesh.nodes[:, 1] == 0))

    # applied load
    area = r * r * np.pi
    ts = [0, -100 * 1e3]
    fs = Matrix.assemble_tractions_over_line(mesh, traction_boundary, ts)

    # check it works, should output zero
    print(fs.sum() - ts[1] * np.pi)
    # lets gooo

    # outer boundaries
    bottom = np.argwhere((mesh.nodes[:, 1] == -depth))
    right = np.argwhere((mesh.nodes[:, 0] == R))
    left = np.argwhere((mesh.nodes[:, 0] == 0))
    top = np.argwhere((mesh.nodes[:, 1] == 0))

    fixed_nodes = np.unique((bottom, right))
    fixed_dof = np.hstack((fixed_nodes, fixed_nodes * 2 + 1))

    E = 20 * 1e6
    nu = 0.3

    K = Matrix.assemble_stiffness_matrix(mesh, E, nu)

    # solving the system
    u_set = fixed_dof * 0
    eq_to_solve = np.setdiff1d(np.arange(mesh.nn * 2), fixed_dof)
    f = -K[eq_to_solve][:, fixed_dof] * u_set + fs[eq_to_solve]

    u_aux = linalg.spsolve(K[eq_to_solve][:, eq_to_solve], f)
    u_res = np.zeros(mesh.nn * 2)
    u_res[eq_to_solve] = u_aux

    udisp = u_res.reshape(2, -1, order='F')
