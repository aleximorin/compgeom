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
    nrad = 80
    theta = np.linspace(0, np.pi/2, nrad)

    x = np.hstack(([0], np.cos(theta) * r))
    y = np.hstack(([0], np.sin(theta) * r))

    vertices = np.vstack((x, y)).T
    edges = Mesher.build_edges(vertices)

    mesh = Mesher.Mesh(vertices, edges, cell_size=1, simultype='axis')

    # ok but we take the matlab mesh lol

    matmesh = loadmat('../../mesh.mat')
    mesh.nodes = matmesh['nodes']
    mesh.connectivity = matmesh['connectivity'] - 1
    mesh.ne = len(mesh.connectivity)
    mesh.nn = len(mesh.nodes)
    mesh.id = np.zeros(mesh.ne).astype(int)

    mesh.plot()

    # geomechanical parameters
    k = 8.4e3  # elastic drained bulk modulus [MPa]
    g = 6.8e3  # shear modulus [MPa]
    b = 0.707692  # biot coefficient
    M = 9.18478e3  # biot modulus [MPa]
    k_u = k + M*b**2
    perm = 0.137549e-3  # permeability
    B = (k_u - k) / (b * k_u)
    mu_f = 1  # fluid viscosity
    rho = 1  # density
    kappa = perm / mu_f  # conductivity
    nu_u = prop.poisson_ratio(g, k_u)  # undrained poisson ratio
    E_u = prop.young_modulus(g, k_u)  # undrained yougn modulus
    nu = prop.poisson_ratio(g, k)  # poisson ratio
    E = prop.young_modulus(g, k)  # young modulus
    eta = b * (1 - 2 * nu_u) / (2 * (1 - nu_u))  # idk what this is

    # boundary conditions
    bottom = np.argwhere(mesh.nodes[:, 1] == 0)
    left = np.argwhere(mesh.nodes[:, 0] < 1e-3)

    fixed_dofx = left * 2
    fixed_dofy = bottom * 2 + 1

    fixed_nodes = np.unique((left, bottom))
    fixed_dof = np.unique((fixed_dofx, fixed_dofy))

    radius = np.linalg.norm(mesh.nodes, axis=1)
    circle = np.argwhere(np.abs(radius - r) <= 1e-3)

    # initial stress field
    sig_unif = [-1, -1, 0, -1]
    sigma0 = Matrix.set_stress_field(mesh, sig_unif)

    # assembling the different matrices
    K = Matrix.assemble_stiffness_matrix(mesh, E, nu)
    Mass = Matrix.assemble_mass_matrix(mesh, 1/M)
    C = Matrix.assemble_conductivity_matrix(mesh, kappa)
    Ce = Matrix.assemble_coupling_matrix(mesh, b)

    