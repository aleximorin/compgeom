import sys
import os

import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import Element

import PoroElasticProperties as prop

print('haha')


def find_eltype(mesh):
    ne, nc = mesh.connectivity.shape
    eltype = 'linear' if nc == 3 else 'quadratic' if nc == 6 else 'undefined'
    return eltype


def assemble_stiffness_matrix(mesh, E, nu):

    K = sparse.lil_matrix((2*mesh.nn, 2*mesh.nn))
    eltype = find_eltype(mesh)

    k = prop.bulk_modulus(E, nu)
    g = prop.shear_modulus(E, nu)

    D = elastic_isotropic_stiffness(k, g, simultype=mesh.simultype)

    for e in range(mesh.ne):

        # access the nodes index of element e
        n_e = mesh.connectivity[e]
        n_dof = np.vstack([2*n_e, 2*n_e + 1]).reshape(-1, order='F')

        # we get the coordinates of the nodes
        X = mesh.nodes[n_e]

        elt = Element.Triangle(X, eltype, mesh.simultype)

        K_el = elt.element_stiffness_matrix(D)

        for i, ni in enumerate(n_dof):
            for j, nj in enumerate(n_dof):
                K[ni, nj] += K_el[i, j]

    return K


def project_flux(mesh, K, head, M=None, return_M=False):

    if sparse.issparse(head):
        head = head.toarray()

    if len(head.shape) == 1:
        head = head[:, None]

    if M is None:
        M = assemble_mass_matrix(mesh, 1.0)

    eltype = find_eltype(mesh)

    if mesh.simultype == '2D':
        f = sparse.csc_matrix((2, mesh.nn))
    else:
        raise ValueError('Not implemented yet')

    for e in range(mesh.ne):

        # access the nodes index of element e
        n_e = mesh.connectivity[e]

        # we get the coordinates of the nodes
        X = mesh.nodes[n_e]

        elt = Element.Triangle(X, eltype, mesh.simultype)

        elt_head = head[n_e]

        f_el = elt.project_element_flux(K, elt_head)
        f[:, n_e] = f[:, n_e] + f_el.T

    f_out = np.zeros_like(f.toarray())
    for i in range(f.shape[0]):
        f_out[i] = linalg.spsolve(M, f[i].T)

    if return_M:
        return f_out, M
    else:
        return f_out


def project_stress(mesh, E, nu, displacement, M=None, return_M=False):
    if M is None:
        M = assemble_mass_matrix(mesh, 1.0)

    eltype = find_eltype(mesh)

    k = prop.bulk_modulus(E, nu)
    g = prop.shear_modulus(E, nu)
    D = elastic_isotropic_stiffness(k, g, mesh.simultype)

    if mesh.simultype == '2D':
        f = sparse.csc_matrix((3, mesh.nn))
    else:
        raise ValueError('Not implemented yet')

    for e in range(mesh.ne):

        # access the nodes index of element e
        n_e = mesh.connectivity[e]
        n_dof = np.vstack([2*n_e, 2*n_e + 1]).reshape(-1, order='F')

        # we get the coordinates of the nodes
        X = mesh.nodes[n_e]

        elt = Element.Triangle(X, eltype, mesh.simultype)

        elt_displacement = displacement[n_dof]

        f_el = elt.project_element_stress(D, elt_displacement)
        f[:, n_e] = f[:, n_e] + f_el.T

    f_out = np.zeros_like(f.toarray())
    for i in range(f.shape[0]):
        f_out[i] = linalg.spsolve(M, f[i].T)

    if return_M:
        return f_out, M
    else:
        return f_out


def elastic_isotropic_stiffness(k, g, simultype='2D'):
    La = k + (4. / 3.) * g
    Lb = k - (2. / 3.) * g

    if simultype == '2D':
        D = np.array([[La, Lb, 0],
                      [Lb, La, 0],
                      [0,   0, g]])

    elif simultype == 'axis':
        D = np.array([[La, Lb, 0, Lb],
                      [Lb, La, 0, Lb],
                      [0,   0, g,  0],
                      [Lb, Lb, 0, La]])

    else:
        raise ValueError('Type not implemented yet')

    return D


def assemble_mass_matrix(mesh, rho):
    M = sparse.csc_matrix((mesh.nn, mesh.nn))
    eltype = find_eltype(mesh)

    if np.isscalar(rho):
        rho = [rho]

    for e in range(mesh.ne):

        # access the nodes index of element e
        n_e = mesh.connectivity[e]

        # we get the coordinates of the nodes
        X = mesh.nodes[n_e]

        # we access the conductivity of the element
        mat_id = mesh.id[e]
        rho_e = rho[mat_id]

        elt = Element.Triangle(X, eltype, mesh.simultype)
        M_el = elt.element_mass_matrix(rho_e)

        for i, ni in enumerate(n_e):
            for j, nj in enumerate(n_e):
                M[ni, nj] += M_el[i, j]

    return M


def assemble_conductivity_matrix(mesh, cond):

    C = sparse.lil_matrix((mesh.nn, mesh.nn))
    eltype = find_eltype(mesh)

    # we want to ensure the conductivity to be accessible by index
    if np.isscalar(cond):
        cond = [cond]

    for e in range(mesh.ne):

        # access the nodes index of element e
        n_e = mesh.connectivity[e]

        # we get the coordinates of the nodes
        X = mesh.nodes[n_e]

        # we access the conductivity of the element
        mat_id = mesh.id[e]
        cond_e = cond[mat_id]

        elt = Element.Triangle(X, eltype, mesh.simultype)
        C_el = elt.element_conductivity_matrix(cond_e)

        for i, ni in enumerate(n_e):
            for j, nj in enumerate(n_e):
                C[ni, nj] += C_el[i, j]

    return C


def assemble_coupling_matrix(mesh, alpha):

    C = sparse.lil_matrix((2*mesh.nn, mesh.nn))
    eltype = find_eltype(mesh)

    for e in range(mesh.ne):

        # access the nodes index of element e
        n_e = mesh.connectivity[e]
        n_dof = np.vstack([2*n_e, 2*n_e + 1]).reshape(-1, order='F')

        # we get the coordinates of the nodes
        X = mesh.nodes[n_e]

        elt = Element.Triangle(X, eltype, mesh.simultype)

        ceel = elt.element_coupling_matrix(alpha)

        for i, ni in enumerate(n_dof):
            for j, nj in enumerate(n_e):
                C[ni, nj] += ceel[i, j]

    return C


def set_stress_field(mesh, stress_field, applied_elements=None):

    S = np.zeros(2*mesh.nn)
    eltype = find_eltype(mesh)

    # we want to find nodes with applied stress
    if applied_elements is None:
        applied_elements = np.arange(mesh.ne)

    il = np.isin(mesh.connectivity, applied_elements)
    elt_line = np.argwhere(il.sum(axis=1) == mesh.connectivity.shape[1])[:, 0]

    for i, e in enumerate(elt_line):

        n_e = mesh.connectivity[e]
        n_dof = np.vstack([2*n_e, 2*n_e + 1]).reshape(-1, order='F')
        X = mesh.nodes[n_e]

        elt = Element.Triangle(X, eltype, mesh.simultype)
        S_el = elt.element_stress_field(stress_field)
        S[n_dof] += S_el

    return S


def assemble_tractions_over_line(mesh, node_list, traction):

    eltype = find_eltype(mesh)

    il = np.isin(mesh.connectivity, node_list)

    n = 2 if eltype == 'linear' else 3
    elt_line = np.argwhere(il.sum(axis=1) == n)[:, 0]  # won't work with quadratic triangle elements, change 2

    f = np.zeros(2*mesh.nn)

    for i, e in enumerate(elt_line):

        nn_l = il[e]

        global_nodes = mesh.connectivity[e, nn_l]
        global_dof = np.array([global_nodes * 2, global_nodes * 2 + 1]).T

        X = mesh.nodes[global_nodes]

        seg_xi, seg_yi = np.argsort(X, axis=0).T
        segx, segy = X[seg_xi, 0], X[seg_yi, 1]

        elt_x = Element.Segment(segx, eltype=eltype, simultype=mesh.simultype)
        elt_y = Element.Segment(segy, eltype=eltype, simultype=mesh.simultype)

        # problem here with neumann and the shape of the segment element
        fs = elt_y.neumann(traction[0])
        fn = elt_x.neumann(traction[1])

        f[global_dof[seg_yi, 0]] += fs
        f[global_dof[seg_xi, 1]] += fn

    return f
