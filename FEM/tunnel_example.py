import numpy as np
import matplotlib.pyplot as plt
import Mesher
import Matrix

import PoroElasticProperties as prop

from scipy.sparse import linalg
from scipy.sparse import lil_matrix

from scipy.io import loadmat

if __name__ == '__main__':

    # problem geometry
    radius = 1
    theta = np.linspace(0, np.pi/2, 10)
    box_length = 50*radius

    vertices = np.vstack((np.cos(theta), np.sin(theta))).T
    other_vertices = np.array([[0, box_length],
                               [box_length, box_length],
                               [box_length, 0]])
    vertices = np.vstack((vertices, other_vertices))

    edges = Mesher.build_edges(vertices)
    mesh = Mesher.Mesh(vertices, edges, cell_size=10)

    matmesh = loadmat('../../matlab_mesh.mat')

    mesh.nodes = matmesh['nodes']
    mesh.connectivity = matmesh['connectivity'] - 1
    mesh.nn = len(mesh.nodes)
    mesh.ne = len(mesh.connectivity)
    mesh.id = np.zeros(mesh.ne).astype(int)

    # we get the boundaries needed to enforce boundary conditions
    left = np.argwhere(np.abs(mesh.nodes[:, 0] - mesh.nodes[:, 0].min()) <= 1e-4)[:, 0]
    right = np.argwhere(np.abs(mesh.nodes[:, 0] - mesh.nodes[:, 0].max()) <= 1e-4)[:, 0]
    bottom = np.argwhere(np.abs(mesh.nodes[:, 1] - mesh.nodes[:, 1].min()) <= 1e-4)[:, 0]
    top = np.argwhere(np.abs(mesh.nodes[:, 1] - mesh.nodes[:, 1].max()) <= 1e-4)[:, 0]
    tunnel = np.argwhere(np.abs(np.linalg.norm(mesh.nodes, axis=1) - radius) <= 1e-4)[:, 0]

    mesh.plot()
    # where do we enforce boundaries
    for boundary in [left, right, bottom, top, tunnel]:
        plt.scatter(*mesh.nodes[boundary].T)

    # we get the corresponding degree of freedom for the boundaries with no displacement
    dof_left = 2*left
    dof_bottom = 2*bottom + 1
    nodes_fixed = np.unique(np.hstack((left, bottom)))
    dof_fixed = np.unique(np.hstack((dof_left, dof_bottom)))

    # we set the initial stress field
    p0 = 30  # mean compressive
    s0 = 10  # mean deviatoric
    stress_field = -np.array([p0-s0, p0+s0, 0])
    sigma = Matrix.set_stress_field(mesh, stress_field)

    fx_right = np.sum(sigma[right * 2])/box_length
    fy_top = np.sum(sigma[top * 2 + 1])/box_length

    # we set the initial pore pressure field, it is zero on the tunnel's boundary
    h0 = 3  # [Mpa]
    pore_pressure_free = np.setdiff1d(np.arange(mesh.nn), tunnel)
    h = np.zeros(mesh.nn)
    h[pore_pressure_free] = h0

    # geomechanical parameters
    k = 8.4e3  # elastic drained bulk modulus [MPa]
    g = 6.8e3  # shear modulus [MPa]
    b = 0.707692  # biot coefficient
    M = 9.18478e3  # biot modulus [MPa]
    k_u = prop.undrained_bulk_modulus(k, b, M)
    perm = 0.137549e-3  # permeability
    B = (k_u-k)/(b*k_u)
    mu_f = 1  # fluid viscosity
    rho = 1  # density
    kappa = perm/mu_f  # conductivity
    nu_u = prop.poisson_ratio(g, k_u)  # undrained poisson ratio
    E_u = prop.young_modulus(g, k_u)  # undrained yougn modulus
    nu = prop.poisson_ratio(g, k)  # poisson ratio
    E = prop.young_modulus(g, k)  # young modulus
    eta = b*(1-2*nu_u)/(2*(1-nu_u))  # idk what this is

    # we build the necessary matrices
    K = Matrix.assemble_stiffness_matrix(mesh, E, nu)
    S = Matrix.assemble_mass_matrix(mesh, 1/M)
    C = Matrix.assemble_conductivity_matrix(mesh, kappa)
    Ce = Matrix.assemble_coupling_matrix(mesh, b)

    # we assemble the force vector
    fx = Matrix.assemble_tractions_over_line(mesh, right, [fx_right, 0])
    fy = Matrix.assemble_tractions_over_line(mesh, top, [0, fy_top])
    f = fx + fy - sigma

    # we can test the system to see if there's any displacement in equilibrium
    f_test = f + Matrix.assemble_tractions_over_line(mesh, tunnel, -stress_field)
    solve = np.setdiff1d(np.arange(2*mesh.nn), dof_fixed)

    u_solved = linalg.inv(K[solve][:, solve]) @ f_test[solve]
    print(f'Maximum displacement should be quite small: {np.abs(u_solved).max():.2e} m')

    # we can solve for the undrained solution, defining the single large matrix
    dt = 0
    AA = S + dt*C

    ntot_E = mesh.nn*2
    ntot_P = mesh.nn
    ntot = ntot_E + ntot_P

    T = lil_matrix((ntot, ntot))
    T[:ntot_E, :ntot_E] = K
    T[ntot_E:, :ntot_E] = -Ce.T
    T[:ntot_E, ntot_E:] = -Ce
    T[ntot_E:, ntot_E:] = -AA

    # we can now build the parts of the equations that we want to solve
    displacement_free = np.setdiff1d(np.arange(ntot_E), dof_fixed)
    eq_free = np.hstack([displacement_free, pore_pressure_free + ntot_E])
    eq_fixed = np.setdiff1d(np.arange(ntot), eq_free)

    # we need to set the force vector to the initial conditions, accounting to pore pressure
    ftot = np.zeros(ntot)
    ftot[:ntot_E] = f - Ce @ h
    ftot[ntot_E:] = -AA @ h

    # we solve the system
    sol_undrained = lil_matrix((ntot, 1))
    sol_undrained[eq_free] = linalg.inv(T[eq_free][:, eq_free]) @ ftot[eq_free]

    # we can separate the solution in both displacement and pore pressure
    displacement_undrained = sol_undrained[:ntot_E].toarray().reshape(2, -1, order='F')
    pore_pressure_undrained = sol_undrained[ntot_E:].toarray()

    flux, M = Matrix.project_flux(mesh, kappa, pore_pressure_undrained, return_M=True)
    du = Matrix.project_stress(mesh, E, nu, sol_undrained[:ntot_E], M=M)

    mesh.plot(displacement_undrained[0])
    mesh.plot(displacement_undrained[1])

    # we want to compute the time varying solution on log-spaced time intervals
    t = np.logspace(-1, 3, 50)
    dt = np.diff(t)

    solutions = np.zeros((len(t), ntot))
    solutions[0] = sol_undrained.toarray()[:, 0]

    # this loop is pretty long, can this be improved? look into sparses matrices operations
    # might also be better to just swap for np arrays?
    for i, dt_i in enumerate(dt):
        # we adjust the total matrix with a varying dt
        T[ntot_E:, ntot_E:] = -(S + dt_i * C)
        # the change in flux is adjusted
        flux = dt_i * C[pore_pressure_free] @ solutions[i, ntot_E:]
        ftot[pore_pressure_free + ntot_E] = flux

        # we can solve the system
        update = linalg.inv(T[eq_free][:, eq_free]) @ ftot[eq_free]
        solutions[i + 1, eq_free] = solutions[i, eq_free] + update

    # we can separate the displacements from the pore pressure to look at it
    displacement_field = solutions[:, :ntot_E].reshape(2, len(t), -1, order='F')
    pore_pressure_field = solutions[:, ntot_E:]

    # we can animate it

    import types
    from matplotlib.animation import ArtistAnimation, FFMpegWriter, FuncAnimation

    fig = plt.figure()
    ax = fig.add_subplot()
    im = ax.tripcolor(*mesh.nodes.T, mesh.connectivity, pore_pressure_field[0], lw=1, shading='gouraud')
    time = ax.set_title('t=0')

    def update(i):
        time.set_text(f't={t[i]:.2f}')
        im.set_array(pore_pressure_field[i])
        return im, time

    ani = FuncAnimation(fig, update, frames=len(t), blit=True, interval=20)
    FFwriter = FFMpegWriter(fps=30, codec='h264', bitrate=100000)
    ani.save(f'anim.mp4', writer=FFwriter)
