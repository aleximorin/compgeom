import numpy as np
from GaussianQuadratures import gaussian_quadrature


class Triangle:

    def __init__(self, X, eltype='linear', simultype='2D'):
        self.X = X
        self.eltype = eltype
        self.simultype = simultype

        if self.simultype not in ('2D', 'axis'):
            raise ValueError('Not implemented yet')

        self.gaussian_quadrature = gaussian_quadrature['triangle'][eltype]

    def N(self, x):
        # x must be of shape (n points, n dimensions)
        # for example x of shape (10, 2) means that it contains ten 2d pointsè

        if len(x.shape) == 1:
            x = np.array(x)  # will this give x the shape (1, n)?

        if self.eltype == 'linear':
            n = np.array([1 - x[:, 0] - x[:, 1],
                          x[:, 0],
                          x[:, 1]]).T

        elif self.eltype == 'quadratic':
            n = np.array([x[:, 0] * (2 * x[:, 0] - 1),
                          x[:, 1] * (2 * x[:, 1] - 1),
                          (1 - x[:, 0] - x[:, 1]) * (2 * (1 - x[:, 0] - x[:, 1]) - 1),
                          4 * x[:, 0] * x[:, 1],
                          4 * x[:, 1] * (1 - x[:, 0] - x[:, 1]),
                          4 * x[:, 0] * (1 - x[:, 0] - x[:, 1])]).T

        else:
            raise ValueError('Not implemented yet')

        return n

    def gradN(self, x):
        # gradient of the shape function

        if self.eltype == 'linear':
            DnaDxi = np.array([[-1, 1, 0],
                               [-1, 0, 1]])
            DnaDxi = np.stack([DnaDxi for i in range(len(x))])

        elif self.eltype == 'quadratic':
            DnaDxi = np.array(
                [[4 * x[:, 0] - 1, 0*x[:, 0], 4 * (x[:, 0] + x[:, 1]) - 3, 4*x[:, 1], -4 * x[:, 1], 4 * (1 - 2*x[:, 0] - x[:, 1])],
                 [0 * x[:, 0], 4 * x[:, 1] - 1, 4 * (x[:, 0] + x[:, 1]) - 3, 4*x[:, 0], 4 * (1 - 2*x[:, 1] - x[:, 0]), - 4 * x[:, 0]]])

            DnaDxi = np.moveaxis(DnaDxi, [2, 1], [0, 2])  # cant find a better way

        else:
            raise ValueError('Not implemented yet')

        J = DnaDxi @ self.X
        j = np.linalg.det(J)
        DxiDx = np.linalg.inv(J)
        DNaDX = DxiDx @ DnaDxi

        return DNaDX, j

    def mapX(self, x):
        return self.N(x) @ self.X

    def element_conductivity_matrix(self, cond):
        xg, wg = self.gaussian_quadrature[1]
        DNaDX, j = self.gradN(xg)
        DNaDXT = np.moveaxis(DNaDX, -1, -2)

        if self.simultype == '2D':
            left = (wg * j * cond)

        elif self.simultype == 'axis':
            mapx = self.mapX(xg)
            left = (2 * np.pi * wg * j * cond * mapx[:, 0])

        else:
            raise ValueError('Not implemented yet')

        # A is (n gauss points, n el, n el)
        A = np.einsum('ijk, ikl -> ijl', DNaDXT, DNaDX)
        kel = np.sum(left[:, None, None] * A, axis=0)

        return kel

    def element_stiffness_matrix(self, D):
        xg, wg = self.gaussian_quadrature[1]
        B, j = self.B_strain_matrix(xg)
        BT = np.moveaxis(B, -1, -2)

        if self.simultype == '2D':
            left = wg * j

        elif self.simultype == 'axis':
            # why do we just take the 0th element??? according to matlab here
            mapx = self.mapX(xg)[:, 0]
            left = 2 * np.pi * mapx * wg * j

            """
            # this would work if we were to not only use the 0th element
            left = 2 * np.pi * mapx * wg * j
            kel = np.einsum('ij, ikl -> jkl', left, BT @ D @ B))
            """

        else:
            raise ValueError('Not implemented yet')

        kel = np.sum(left * BT @ D @ B, axis=0)

        return kel

    def element_mass_matrix(self, rho):
        xg, wg = self.gaussian_quadrature[2]

        DnaDx, j = self.gradN(xg)
        N_i = self.N(xg)

        if self.simultype == '2D':
            left = (wg * j * rho)

        elif self.simultype == 'axis':
            mapx = self.mapX(xg)
            left = 2 * np.pi * wg * j * rho * mapx[:, 0]

        else:
            raise ValueError('Not implemented yet')

        A = np.einsum('ij, ik -> ijk', N_i, N_i)
        kel = np.sum(left[:, None, None] * A, axis=0)

        return kel

    def B_strain_matrix(self, xg):
        DnaDx, j = self.gradN(xg)
        zero = 0 * DnaDx[:, 0, 0]

        if self.simultype == '2D':
            if self.eltype == 'linear':
                # 3 x 6 matrix
                B = np.array([[DnaDx[:, 0, 0], zero, DnaDx[:, 0, 1], zero, DnaDx[:, 0, 2], zero],
                             [zero, DnaDx[:, 1, 0], zero, DnaDx[:, 1, 1], zero, DnaDx[:, 1, 2]],
                             [DnaDx[:, 1, 0], DnaDx[:, 0, 0], DnaDx[:, 1, 1], DnaDx[:, 0, 1], DnaDx[:, 1, 2], DnaDx[:, 0, 2]]])

            elif self.eltype == 'quadratic':
                # 3 x 12 matrix
                B = np.array([[DnaDx[:, 0, 0], zero, DnaDx[:, 0, 1], zero, DnaDx[:, 0, 2], zero,
                               DnaDx[:, 0, 3], zero, DnaDx[:, 0, 4], zero, DnaDx[:, 0, 5], zero],
                              [zero, DnaDx[:, 1, 0], zero, DnaDx[:, 1, 1], zero, DnaDx[:, 1, 2],
                               zero, DnaDx[:, 1, 3], zero, DnaDx[:, 1, 4], zero, DnaDx[:, 1, 5]],
                              [DnaDx[:, 1, 0], DnaDx[:, 0, 0], DnaDx[:, 1, 1], DnaDx[:, 0, 1], DnaDx[:, 1, 2], DnaDx[:, 0, 2],
                               DnaDx[:, 1, 3], DnaDx[:, 0, 3], DnaDx[:, 1, 4], DnaDx[:, 0, 4], DnaDx[:, 1, 5], DnaDx[:, 0, 5]]])

        elif self.simultype == 'axis':

            mapx = self.mapX(xg)
            Nx = self.N(xg)

            if self.eltype == 'linear':
                # 4 x 6 matrix
                B = np.array([[DnaDx[:, 0, 0], zero, DnaDx[:, 0, 1], zero, DnaDx[:, 0, 2], zero],
                              [zero, DnaDx[:, 1, 0], zero, DnaDx[:, 1, 1], zero, DnaDx[:, 1, 2]],
                              [DnaDx[:, 1, 0], DnaDx[:, 0, 0], DnaDx[:, 1, 1], DnaDx[:, 0, 1], DnaDx[:, 1, 2], DnaDx[:, 0, 2]],
                              [Nx[:, 0] / mapx[:, 0], zero, Nx[:, 1] / mapx[:, 0], zero, Nx[:, 0] / mapx[:, 0], zero]])

            if self.eltype == 'quadratic':
                # don't know if it works, will need to check with a quadratic mesh
                # 4 x 12 matrix
                B = np.array([[DnaDx[:, 0, 0], zero, DnaDx[:, 0, 1], zero, DnaDx[:, 0, 2], zero,
                               DnaDx[:, 0, 3], zero, DnaDx[:, 0, 4], zero, DnaDx[:, 0, 5], zero],
                              [zero, DnaDx[:, 1, 0], zero, DnaDx[:, 1, 1], zero, DnaDx[:, 1, 2],
                               zero, DnaDx[:, 1, 3], zero, DnaDx[:, 1, 4], zero, DnaDx[:, 1, 5]],
                              [DnaDx[:, 1, 0], DnaDx[:, 0, 0], DnaDx[:, 1, 1], DnaDx[:, 0, 1], DnaDx[:, 1, 2], DnaDx[:, 0, 2],
                               DnaDx[:, 1, 3], DnaDx[:, 0, 3], DnaDx[:, 1, 4], DnaDx[:, 0, 4], DnaDx[:, 1, 5], DnaDx[:, 0, 5]],
                              [Nx[:, 0]/mapx[:, 0], zero, Nx[:, 1]/mapx[:, 0], zero, Nx[:, 2]/mapx[:, 0], zero,
                               Nx[:, 3]/mapx[:, 0], zero, Nx[:, 4]/mapx, zero, Nx[:, 5]/mapx, zero]])


        else:
            raise ValueError('Not implemented yet')

        B = np.moveaxis(B, -1, 0)
        return B, j

    def element_coupling_matrix(self, alpha):
        xg, wg = self.gaussian_quadrature[2]
        B, j = self.B_strain_matrix(xg)
        N_i = self.N(xg)

        if self.simultype == '2D':
            Baux = B[:, 0] + B[:, 1]
            left = wg * j * alpha

        elif self.simultype == 'axis':
            Baux = B[:, 0] + B[:, 1] + B[:, -1]
            mapx = self.mapX(xg)
            left = 2 * np.pi * wg * j *alpha * mapx[:, 0]

        else:
            raise ValueError('Not implemented yet')

        A = np.einsum('ij, ik -> ijk', Baux, N_i)
        ceel = np.sum(left[:, None, None] * A, axis=0)

        return ceel

    def element_stress_field(self, stress_field):
        xg, wg = self.gaussian_quadrature[2]
        B, j = self.B_strain_matrix(xg)
        BT = np.moveaxis(B, -1, -2)

        if self.simultype == '2D':
            left = j * wg

        elif self.simultype == 'axis':
            mapx = self.mapX(xg)
            left = 2 * np.pi * j * wg * mapx[:, 0]

        else:
            raise ValueError('Not implemented yet')

        A = BT @ stress_field
        s_el = left @ A

        return s_el

    def project_element_stress(self, D, displacement):
        xg, wg = self.gaussian_quadrature[2]

        if self.simultype == '2D':
            N_i = self.N(xg)
            B, j = self.B_strain_matrix(xg)

            # we initially solve for the displacement
            A = np.einsum('ijk, kl -> ij', B, displacement)
            solve = np.einsum('ij, ki -> kj', D, A)

            # it is simpler for broadcasting to multiply j_i and wg_i together
            left = j * wg
            f_el = np.sum(left[:, None, None] * np.einsum('ij, ik -> ijk', N_i, solve), axis=0)
        else:
            raise ValueError('Not implemented yet')

        return f_el

    def project_element_flux(self, K, head):
        xg, wg = self.gaussian_quadrature[2]

        if self.simultype == '2D':
            N_i = self.N(xg)
            DNaDx, j = self.gradN(xg)
            solve = - K * np.einsum('ijk, kl -> ij', DNaDx, head)
            left = j * wg

        elif self.simultype == 'axis':
            raise ValueError('Not implemented yet')

        else:
            raise ValueError('Not implemented yet')

        f_el = np.sum(left[:, None, None] * np.einsum('ij, ik -> ijk', N_i, solve), axis=0)

        return f_el


class Segment:
    def __init__(self, X, eltype='linear', simultype='2D'):
        self.X = X
        self.eltype = eltype
        self.simultype = simultype

        if self.simultype not in ('2D', 'axis'):
            raise ValueError('Not implemented yet')

        self.gaussian_quadrature = gaussian_quadrature['segment'][eltype]

    def N(self, x):
        # x must be of shape (n points, n dimensions)
        # for example x of shape (10, 2) means that it contains ten 2d points

        if self.eltype == 'linear':
            n = np.array([0.5*(1 - x), 0.5*(1 + x)]).T
        elif self.eltype == 'quadratic':
            n = np.array([0.5 * x * (x - 1), (1 + x) * (1 - x), 0.5 * x * (x + 1)]).T
        else:
            raise ValueError('Not implemented yet')

        return n

    def gradN(self, x):
        # gradient of the shape function
        # DN has shape (n gauss points, n segment points)

        if self.eltype == 'linear':
            DN = np.array([-0.5, 0.5])[None, :]

        elif self.eltype == 'quadratic':
            DN = np.array([x - 0.5, -2 * x, x + 0.5]).T  # x has len 3, DN has shape (3, 3), self.X has shape 3

        else:
            raise ValueError('Not implemented yet')

        # j then has shape (n gauss points,)
        j = DN @ self.X
        DNaDX = DN/j

        return DNaDX, j

    def mapX(self, x):
        return self.N(x) @ self.X

    def B_strain_matrix(self, x):

        if self.simultype == '2D':
            DN = self.gradN(x)
            return DN

        else:
            raise ValueError('Not implemented yet')

    def neumann(self, f):
        xg, wg = self.gaussian_quadrature[1]
        DNaDX, j = self.gradN(xg)
        N = self.N(xg)

        if self.simultype == '2D':
            left = wg * j

        elif self.simultype == 'axis':
            mapx = self.mapX(xg)
            left = 2 * np.pi * wg * j * mapx

        else:
            raise ValueError('Not implemented yet')

        fel = np.sum(left[:, None] * N * f, axis=0)

        return fel
