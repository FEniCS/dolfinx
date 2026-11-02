"""Analytical fields and small UFL helpers."""

import numpy as np
from scipy.special import h2vp, hankel2, jv, jvp

import ufl
from dolfinx import fem


def compute_a(nu: int, m: complex, alpha: float) -> complex:
    J_nu_alpha = jv(nu, alpha)
    J_nu_malpha = jv(nu, m * alpha)
    J_nu_alpha_p = jvp(nu, alpha, 1)
    J_nu_malpha_p = jvp(nu, m * alpha, 1)

    H_nu_alpha = hankel2(nu, alpha)
    H_nu_alpha_p = h2vp(nu, alpha, 1)

    a_nu_num = J_nu_alpha * J_nu_malpha_p - m * J_nu_malpha * J_nu_alpha_p
    a_nu_den = H_nu_alpha * J_nu_malpha_p - m * J_nu_malpha * H_nu_alpha_p
    return a_nu_num / a_nu_den


def calculate_analytical_efficiencies(
    eps: complex, n_bkg: float, wl0: float, radius_wire: float, num_n: int = 50
) -> tuple[float, float, float]:
    m = np.sqrt(np.conj(eps)) / n_bkg
    alpha = 2 * np.pi * radius_wire / wl0 * n_bkg
    c = 2 / alpha
    q_ext = c * np.real(compute_a(0, m, alpha))
    q_sca = c * np.abs(compute_a(0, m, alpha)) ** 2

    for nu in range(1, num_n + 1):
        q_ext += c * 2 * np.real(compute_a(nu, m, alpha))
        q_sca += c * 2 * np.abs(compute_a(nu, m, alpha)) ** 2

    return q_ext - q_sca, q_sca, q_ext


def background_field(theta: float, n_b: float, k0: complex, x: np.ndarray):
    kx = n_b * k0 * np.cos(theta)
    ky = n_b * k0 * np.sin(theta)
    phi = kx * x[0] + ky * x[1]
    return (-np.sin(theta) * np.exp(1j * phi), np.cos(theta) * np.exp(1j * phi))


def curl_2d(a: fem.Function):
    return ufl.as_vector((0, 0, a[1].dx(0) - a[0].dx(1)))
