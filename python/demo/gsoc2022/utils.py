import numpy as np
from ufl import as_vector
from dolfinx import io
from mpi4py import MPI


def save_as_xdmf(filepath, mesh, E):

    with io.XDMFFile(MPI.COMM_WORLD, filepath, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(E)


class background_electric_field:

    def __init__(self, theta, n_bkg, k0):
        self.theta = theta
        self.k0 = k0
        self.n_bkg = n_bkg

    def eval(self, x):

        kx = self.n_bkg * self.k0 * np.cos(self.theta)
        ky = self.n_bkg * self.k0 * np.sin(self.theta)
        phi = kx * x[0] + ky * x[1]

        ax = np.sin(self.theta)
        ay = np.cos(self.theta)

        return (-ax * np.exp(1j * phi), ay * np.exp(1j * phi))


def radial_distance(x):
    return np.sqrt(x[0]**2 + x[1]**2)


def curl_2d(a):

    ay_x = a[1].dx(0)
    ax_y = a[0].dx(1)

    c = as_vector((0, 0, ay_x - ax_y))

    return c


def from_2d_to_3d(a):

    return as_vector((a[0], a[1], 0))


def calculateAnalyticalEfficiencies(reps, ieps, n_bkg, wl0, radius_wire):

    from scipy.special import jv, hankel2, jvp, h2vp

    def a_coeff(n, m, alpha):

        J_n_alpha = jv(n, alpha)
        J_n_malpha = jv(n, m * alpha)
        J_n_alpha_p = jvp(n, alpha, 1)
        J_n_malpha_p = jvp(n, m * alpha, 1)

        H_n_alpha = hankel2(n, alpha)
        H_n_alpha_p = h2vp(n, alpha, 1)

        an_num = J_n_alpha * J_n_malpha_p - m * J_n_malpha * J_n_alpha_p
        an_den = H_n_alpha * J_n_malpha_p - m * J_n_malpha * H_n_alpha_p

        an = an_num / an_den

        return an

    m = np.sqrt(reps - 1j * ieps) / n_bkg

    alpha = 2 * np.pi * radius_wire / wl0 * n_bkg
    c = 2 / alpha

    num_n = 50

    for n in range(0, num_n + 1):

        if n == 0:

            q_ext = c * np.real(a_coeff(n, m, alpha))
            q_sca = c * np.abs(a_coeff(n, m, alpha))**2

        else:

            q_ext += c * 2 * np.real(a_coeff(n, m, alpha))
            q_sca += c * 2 * np.abs(a_coeff(n, m, alpha))**2

    q_abs = q_ext - q_sca

    return q_abs, q_sca, q_ext
