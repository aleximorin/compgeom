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
    cell_size = 0.01

    import pygmsh

    with pygmsh.geo.Geometry() as geom:
        center = geom.add_point([0, 0], 10 * cell_size)
        lower_right = geom.add_point([0, r], cell_size)
        upper_left = geom.add_point([r, 0], cell_size)

        bottom = geom.add_line(center, lower_right)
        left = geom.add_line(upper_left, center)
        arc = geom.add_circle_arc(lower_right, center, upper_left)

        loop = geom.add_curve_loop([bottom, arc, left])
        surface = geom.add_plane_surface(loop)
        out = geom.generate_mesh()

    mesh = Mesher.Mesh(out, simultype='axis')

    from scipy.io import loadmat

    matmesh = loadmat('../../mesh.mat')
    mesh.nodes = matmesh['nodes']
    mesh.connectivity = matmesh['connectivity'] - 1
    mesh.nn = len(mesh.nodes)
    mesh.ne = len(mesh.connectivity)

    mesh.plot()

    #mesh = Mesher.tri3_2_tri6(mesh)

    mesh.plot()
    plt.xlabel('x')
    plt.ylabel('y')
    print(mesh.nn)
    print(mesh.nodes.shape)

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

    # we can now solve the undrained case
    dt = 0
    AA = -Mass - dt *C

    ntot_E = mesh.nn * 2
    ntot_P = mesh.nn
    ntot = ntot_E + ntot_P

    T = csc_matrix((ntot, ntot))
    T[:ntot_E, :ntot_E] = K
    T[ntot_E:, :ntot_E] = -Ce.T
    T[:ntot_E, ntot_E:] = -Ce
    T[ntot_E:, ntot_E:] = -AA

    ftot = csc_matrix((ntot, 1))
    ftot[:ntot_E] = sigma0

    eq_free_disp = np.setdiff1d(np.arange(ntot_E), fixed_dof)
    eq_free_p = np.arange(ntot_P) + ntot_E
    eq_free = np.hstack(eq_free_disp, eq_free_p)

    # solving the system
    undrained_sol = csc_matrix((ntot, 1))
    undrained_sol[eq_free] = linalg.spsolve(T[eq_free][:, eq_free], ftot[eq_free])

    pressure_undrained = undrained_sol[ntot_E:].toarray()
    disp_undrained = undrained_sol[:ntot_E].toarray().reshape(2, -1, order='F')

    