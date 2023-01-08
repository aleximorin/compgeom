import numpy as np
import matplotlib.pyplot as plt
import Mesher
import Matrix

import PoroElasticProperties as prop

from scipy.sparse import linalg
from scipy.sparse import lil_matrix, csc_matrix

from scipy.io import loadmat

plt.rcParams['text.usetex'] = True

if __name__ == '__main__':

    # We define the geometry of the problem
    frac_length = 1
    box_length = 20 * frac_length

    npoints = 20
    vertices = np.vstack((np.linspace(0, 2 * frac_length, 2 * npoints + 1), np.zeros(2 * npoints + 1))).T
    other_vertices = np.array([[box_length, 0],
                               [box_length, box_length],
                               [0, box_length],
                               ])

    vertices = np.vstack((vertices, other_vertices, np.roll(vertices[1:], shift=1, axis=1)[::-1]))

    edges = Mesher.build_edges(vertices)

    plt.scatter(*vertices.T)
    for e in edges:
        (x0, y0), (x1, y1) = vertices[e]
        plt.plot([x0, x1], [y0, y1])
    plt.title('Nodes and edges')

    mesh = Mesher.Mesh(vertices, edges, cell_size=3)
    for r in np.arange(10, 0, -1):
        print(r, r / 10)
        mesh.fan_refine([frac_length, 0], r * frac_length, 0, np.pi, area=r / 10)
    bbox = np.array([[0, 0],
                     [0, frac_length],
                     [frac_length, frac_length],
                     [frac_length, 0]])

    fig, axs = plt.subplots(1, 2, sharex='row', sharey='row', figsize=(10, 4))
    mesh.plot(ax=axs[0])
    axs[0].set_title('Non-refined mesh')
    mesh.refine(10 * bbox, area=1)
    mesh.refine(8 * bbox, area=0.5)
    mesh.refine(5 * bbox, area=0.25)
    mesh.refine(2 * bbox, area=0.1)
    mesh.refine(1 * bbox, area=0.1)
    mesh.plot(ax=axs[1])
    axs[1].set_title('Refined mesh')
    axs[0].set_aspect(1)
    axs[1].set_aspect(1)
    print(f'Refined mesh contains {len(mesh.nodes)} nodes')

    mesh = Mesher.tri3_2_tri6(mesh)
    """matmesh = loadmat('../quad.mat')
    mesh.nodes = matmesh['nodes']
    mesh.connectivity = matmesh['connectivity'] - 1
    mesh.nn = len(mesh.nodes)
    mesh.ne = len(mesh.connectivity)
    mesh.id = np.zeros(mesh.ne).astype(int)"""

    # we get the boundaries needed to enforce boundary conditions
    left = np.argwhere(np.abs(mesh.nodes[:, 0] - mesh.nodes[:, 0].min()) <= 1e-4)[:, 0]
    right = np.argwhere(np.abs(mesh.nodes[:, 0] - mesh.nodes[:, 0].max()) <= 1e-4)[:, 0]
    top = np.argwhere(np.abs(mesh.nodes[:, 1] - mesh.nodes[:, 1].max()) <= 1e-4)[:, 0]

    bottom = np.abs(mesh.nodes[:, 1] - mesh.nodes[:, 1].min()) <= 1e-4
    frac = mesh.nodes[:, 0] < frac_length
    bottom_frac = np.argwhere(bottom & frac)[:, 0]
    bottom_clear = np.argwhere(bottom & ~frac)[:, 0]

    mesh.plot()
    for cond in (left, right, top, bottom_frac, bottom_clear):
        plt.scatter(*mesh.nodes[cond].T, alpha=0.5)

    plt.title('Different boundaries of the mesh')
    plt.gca().set_aspect(1)

    # some geomechanical properties, don't really know what's needed or not
    ## would need to tweak the parameters a bit probably
    k = 8.4e3  # elastic drained bulk modulus [MPa]
    g = 6.8e3  # shear modulus [MPa]
    b = 0.707692  # biot coefficient
    M = 9.18478e3  # biot modulus [MPa]
    k_u = prop.undrained_bulk_modulus(k, b, M)
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

    # Assembly of the different matrices
    K = Matrix.assemble_stiffness_matrix(mesh, E, nu)
    S = Matrix.assemble_mass_matrix(mesh, 1 / M)  # storage is wrong
    C = Matrix.assemble_conductivity_matrix(mesh, kappa)  #
    Ce = Matrix.assemble_coupling_matrix(mesh, b)

    # defining the single large matrix
    dt = 0
    AA = S + dt * C

    ntot_E = mesh.nn * 2
    ntot_P = mesh.nn
    ntot = ntot_E + ntot_P

    T = csc_matrix((ntot, ntot))
    T[:ntot_E, :ntot_E] = K
    T[ntot_E:, :ntot_E] = -Ce.T
    T[:ntot_E, ntot_E:] = -Ce
    T[ntot_E:, ntot_E:] = -AA

    # there won't be any vertical displacement for the nodes after the fracture
    # we get the corresponding degree of freedom for the boundaries with no displacement
    dof_left = 2 * left
    dof_bottom = 2 * bottom_clear + 1
    nodes_fixed = np.unique(np.hstack((left, bottom_clear)))
    dof_fixed = np.unique(np.hstack((dof_left, dof_bottom)))

    # we set the initial stress field
    p0 = 0  # mean compressive
    h0 = 0  # [Mpa]
    h_frac = 100  # [MPa]

    stress_field = -np.array([p0, p0, 0])
    sigma = Matrix.set_stress_field(mesh, stress_field)
    fx_right = np.sum(sigma[right * 2]) / box_length
    fy_top = np.sum(sigma[top * 2 + 1]) / box_length

    # we assemble the force vector
    fx = Matrix.assemble_tractions_over_line(mesh, right, [fx_right, 0])
    fy = Matrix.assemble_tractions_over_line(mesh, top, [0, fy_top])
    ffrac = Matrix.assemble_tractions_over_line(mesh, bottom_frac, [h_frac, h_frac])
    f = fx + fy - sigma + ffrac

    # we set the initial pore pressure field
    #pore_pressure_free = np.setdiff1d(np.arange(mesh.nn), bottom_frac)  # uncomment for permeable fracture
    pore_pressure_free = np.arange(mesh.nn)  # uncomment for unpermeable fracture
    pore_pressure_fixed = np.setdiff1d(np.arange(mesh.nn), pore_pressure_free)
    h = np.zeros(mesh.nn)
    h[pore_pressure_fixed] = h_frac
    h[pore_pressure_free] = h0

    # we can now build the parts of the equations that we want to solve
    displacement_free = np.setdiff1d(np.arange(ntot_E), dof_fixed)
    eq_free = np.hstack([displacement_free, pore_pressure_free + ntot_E])
    eq_fixed = np.setdiff1d(np.arange(ntot), eq_free)

    # setting the force vector to the initial conditions according to pore pressure
    ftot = csc_matrix((ntot, 1))
    ftot[:ntot_E] = f - Ce @ h
    ftot[ntot_E:] = -AA @ h

    # we solve the system
    sol_undrained = csc_matrix((ntot, 1))
    sol_undrained[eq_free] = linalg.inv(T[eq_free][:, eq_free]) @ ftot[eq_free]
    sol_undrained[pore_pressure_fixed + ntot_E] = h_frac

    displacement = sol_undrained.toarray()[:ntot_E, 0].reshape(2, -1, order='F')
    pressure = sol_undrained.toarray()[ntot_E:, 0]

    #mesh.plot(displacement[1])
    # we want to look at the undrained behavior of the problem
    du = Matrix.project_stress(mesh, E_u, nu_u, sol_undrained[:ntot_E].toarray())
    dp = Matrix.project_flux(mesh, kappa, sol_undrained[ntot_E:].toarray())

    # here we plot the many different fields
    fig, axs = plt.subplots(2, 4, figsize=(16, 6), sharex='all', sharey='all')

    _, _, cb = mesh.plot(displacement[0], ax=axs[0, 0])
    cb.ax.set_title('$u_x$ (m)', rotation=0, ha='center', va='bottom')

    _, _, cb = mesh.plot(displacement[1], ax=axs[1, 0])
    cb.ax.set_title('$u_y$ (m)', rotation=0, ha='center', va='bottom')

    _, _, cb = mesh.plot(du[0], ax=axs[0, 1])
    cb.ax.set_title('$\sigma_{xx}$ (MPa)', rotation=0, ha='center', va='bottom')

    _, _, cb = mesh.plot(du[1], ax=axs[1, 1])
    cb.ax.set_title('$\sigma_{yy}$ (MPa)', rotation=0, ha='center', va='bottom')

    _, _, cb = mesh.plot(du[2], ax=axs[0, 2])
    cb.ax.set_title('$\sigma_{xy}$ (MPa)', rotation=0, ha='center', va='bottom')

    _, _, cb = mesh.plot(pressure, ax=axs[1, 2])
    cb.ax.set_title('$p$ (MPa)', rotation=0, ha='center', va='bottom')

    _, _, cb = mesh.plot(dp[0], ax=axs[0, 3])
    cb.ax.set_title('$q_x$ (m s$^{-1}$)', rotation=0, ha='center', va='bottom')

    _, _, cb = mesh.plot(dp[1], ax=axs[1, 3])
    cb.ax.set_title('$q_y$ (m s$^{-1}$)', rotation=0, ha='center', va='bottom')

    axs[0, 0].set_xlim(0, 5 * frac_length)
    axs[0, 0].set_ylim(0, 5 * frac_length)
    [ax.set_aspect(1) for ax in axs.ravel()]
    [ax.set_xlabel('x (m)') for ax in axs[-1, :]]
    [ax.set_ylabel('y (m)') for ax in axs[:, 0]]
    fig.subplots_adjust(hspace=0.15, wspace=0.05)

    # we plot the pressure and displacement profile along the fracture
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    X = mesh.nodes[bottom, 0]
    ii = np.argsort(X)
    l1, = ax.plot(X[ii], pressure[bottom][ii])
    l2, = ax2.plot(X[ii], displacement[0, bottom][ii], c='tab:orange')
    l3, = ax2.plot(X[ii], displacement[1, bottom][ii], c='tab:green')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('p (MPa)')
    ax2.set_ylabel('$u_i$ (m)')
    ax.legend((l1, l2, l3), ('p', '$u_x$', '$u_y$'))
    ax.axvline(1, ls='dashed', lw=1, c='k', zorder=0)

    # we plot the different stress derivatives along the fracture
    fig, ax = plt.subplots()
    X = mesh.nodes[bottom, 0]
    ii = np.argsort(X)
    l1, = ax.plot(X[ii], du[0][bottom][ii])
    l2, = ax.plot(X[ii], du[1][bottom][ii])
    l3, = ax.plot(X[ii], du[2][bottom][ii])

    ax.set_xlabel('x (m)')
    ax.set_ylabel('$\sigma_ii$ (MPa)')
    ax.legend((l1, l2, l3), ('$\sigma_{xx}$', '$\sigma_{yy}$', '$\sigma_{xy}$'))
    ax.axvline(1, ls='dashed', lw=1, c='k', zorder=0)

    # we want to compute the time varying solution on log-spaced time intervals
    t = np.logspace(-4, 6, 100)
    dt = np.diff(t)

    solutions = np.zeros((len(t), ntot))
    solutions[0] = sol_undrained.toarray()[:, 0]

    ftot = csc_matrix((ntot, 1))

    from time import time

    t0 = time()
    for i, dt_i in enumerate(dt):
        print(
            f'\rSolving {i + 1}/{len(dt)} = {(i + 1) / len(dt) * 100:.2f}%. Total computation time: {time() - t0:.2f} s',
            end='')
        # we adjust the total matrix with a varying dt
        T[ntot_E:, ntot_E:] = -(S + dt_i * C)

        # the change in flux is adjusted
        flux = dt_i * C[pore_pressure_free] @ solutions[i, ntot_E:]
        ftot[pore_pressure_free + ntot_E] = flux

        # we can solve the system
        update = linalg.inv(T[eq_free][:, eq_free]) @ ftot[eq_free]
        solutions[i + 1, eq_free] = solutions[i, eq_free] + update.toarray()[:, 0]
        solutions[i + 1, pore_pressure_fixed + ntot_E] = h_frac

    print('\nDing!')
    # we can separate the displacements from the pore pressure to look at it
    displacement_field = solutions[:, :ntot_E].reshape(len(t), 2, -1, order='F')
    pore_pressure_field = solutions[:, ntot_E:]

    du = np.zeros((len(t), 3, mesh.nn))
    dp = np.zeros((len(t), 2, mesh.nn))
    for i in range(len(t)):
        print(solutions[i, :ntot_E].shape)
        du[i] = Matrix.project_stress(mesh, E, nu, solutions[i, :ntot_E][:, None])
        dp[i] = Matrix.project_flux(mesh, kappa, solutions[i, ntot_E:])

    # we can separate the displacements from the pore pressure to look at it
    displacement_field = solutions[:, :ntot_E].reshape(2, len(t), -1, order='F')
    pore_pressure_field = solutions[:, ntot_E:]

    # does it make sense at the fracture?
    x = mesh.nodes[bottom, 0]
    ii = x.argsort()
    p = pore_pressure_field[:, bottom]

    plt.figure()
    plt.pcolormesh(x[ii], t, p[:, ii], shading='gouraud')
    plt.yscale('log')
    plt.colorbar()

    u_y = displacement_field[1][:, bottom]
    plt.figure()
    plt.pcolormesh(x[ii], t, u_y[:, ii], shading='gouraud')
    plt.yscale('log')
    plt.colorbar()

    # we can animate it
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    fig = plt.figure()
    ax = fig.add_subplot()
    im = ax.tripcolor(*mesh.nodes.T, mesh.connectivity, pore_pressure_field[0], lw=1, shading='gouraud')
    time = ax.set_title('t=0')
    #ax.set_xlim(0, 2*frac_length)
    #ax.set_ylim(0, 2*frac_length)

    def update(i):
        time.set_text(f't={t[i]:.2f}')
        im.set_array(pore_pressure_field[i])
        return im, time

    ani = FuncAnimation(fig, update, frames=len(t), blit=True, interval=20)
    FFwriter = FFMpegWriter(fps=30, codec='h264', bitrate=100000)
    ani.save(f'anim_frac.mp4', writer=FFwriter)

    plt.show()

