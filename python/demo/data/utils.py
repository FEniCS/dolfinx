import numpy as np
from scipy.special import h2vp, hankel2, jv, jvp

def calculate_analytical_efficiencies(reps, ieps, n_bkg, wl0, radius_wire):

    def a_coeff(nu, m, alpha):

        J_nu_alpha = jv(nu, alpha)
        J_nu_malpha = jv(nu, m * alpha)
        J_nu_alpha_p = jvp(nu, alpha, 1)
        J_nu_malpha_p = jvp(nu, m * alpha, 1)

        H_nu_alpha = hankel2(nu, alpha)
        H_nu_alpha_p = h2vp(nu, alpha, 1)

        a_nu_num = J_nu_alpha * J_nu_malpha_p - m * J_nu_malpha * J_nu_alpha_p
        a_nu_den = H_nu_alpha * J_nu_malpha_p - m * J_nu_malpha * H_nu_alpha_p

        return a_nu_num / a_nu_den

    m = np.sqrt(reps - 1j * ieps) / n_bkg

    alpha = 2 * np.pi * radius_wire / wl0 * n_bkg
    c = 2 / alpha

    num_n = 50

    for nu in range(0, num_n + 1):

        if nu == 0:

            q_ext = c * np.real(a_coeff(nu, m, alpha))
            q_sca = c * np.abs(a_coeff(nu, m, alpha))**2

        else:

            q_ext += c * 2 * np.real(a_coeff(nu, m, alpha))
            q_sca += c * 2 * np.abs(a_coeff(nu, m, alpha))**2

    q_abs = q_ext - q_sca

    return q_abs, q_sca, q_ext