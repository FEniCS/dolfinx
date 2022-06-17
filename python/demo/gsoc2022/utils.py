import numpy as np
from mpi4py import MPI
from scipy.special import h2vp, hankel2, jv, jvp
from ufl import as_vector

from dolfinx import io


def calculate_analytical_efficiencies(reps, ieps, n_bkg, wl0, radius_wire):

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
