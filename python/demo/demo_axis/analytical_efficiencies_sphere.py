# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Calculation of analytical efficiencies
#
# Copyright (C) 2022 Michele Castriotta, Igor Baratta, Jørgen S. Dokken
#
# This file contains a function for the calculation of the
# absorption, scattering and extinction efficiencies of a wire
# being hit normally by a TM-polarized electromagnetic wave.
#
# The formula are taken from:
# Milton Kerker, "The Scattering of Light and Other Electromagnetic Radiation",
# Chapter 6, Elsevier, 1969.
#
# ## Implementation
# First of all, let's define the parameters of the problem:
#
# - $n = \sqrt{\varepsilon}$: refractive index of the wire,
# - $n_b$: refractive index of the background medium,
# - $m = n/n_b$: relative refractive index of the wire,
# - $\lambda_0$: wavelength of the electromagnetic wave,
# - $r_w$: radius of the cross-section of the wire,
# - $\alpha = 2\pi r_w n_b/\lambda_0$.
#
# Now, let's define the $a_\nu$ coefficients as:
#
# $$
# \begin{equation}
# a_\nu=\frac{J_\nu(\alpha) J_\nu^{\prime}(m \alpha)-m J_\nu(m \alpha)
# J_\nu^{\prime}(\alpha)}{H_\nu^{(2)}(\alpha) J_\nu^{\prime}(m \alpha)
# -m J_\nu(m \alpha) H_\nu^{(2){\prime}}(\alpha)}
# \end{equation}
# $$
#
# where:
# - $J_\nu(x)$: $\nu$-th order Bessel function of the first kind,
# - $J_\nu^{\prime}(x)$: first derivative with respect to $x$ of
# the $\nu$-th order Bessel function of the first kind,
# - $H_\nu^{(2)}(x)$: $\nu$-th order Hankel function of the second kind,
# - $H_\nu^{(2){\prime}}(x)$: first derivative with respect to $x$ of
# the $\nu$-th order Hankel function of the second kind.
#
# We can now calculate the scattering, extinction and absorption
# efficiencies as:
#
# $$
# \begin{align}
# & q_{\mathrm{sca}}=(2 / \alpha)\left[\left|a_0\right|^{2}
# +2 \sum_{\nu=1}^{\infty}\left|a_\nu\right|^{2}\right] \\
# & q_{\mathrm{ext}}=(2 / \alpha) \operatorname{Re}\left[ a_0
# +2 \sum_{\nu=1}^{\infty} a_\nu\right] \\
# & q_{\mathrm{abs}} = q_{\mathrm{ext}} - q_{\mathrm{sca}}
# \end{align}
# $$

from scattnlay import scattnlay
import numpy as np
from scipy.special import spherical_jn, spherical_yn
from typing import Tuple


# The functions that we import from `scipy.special` correspond to:
#
# - `jv(nu, x)` ⟷ $J_\nu(x)$,
# - `jvp(nu, x, 1)` ⟷ $J_\nu^{\prime}(x)$,
# - `hankel2(nu, x)` ⟷ $H_\nu^{(2)}(x)$,
# - `h2vp(nu, x, 1)` ⟷ $H_\nu^{(2){\prime}}(x)$.
#
# Next, we define a function for calculating the analytical efficiencies
# in Python. The inputs of the function are:
#
# - `eps` ⟷ $\varepsilon$,
# - `n_bkg` ⟷ $n_b$,
# - `wl0` ⟷ $\lambda_0$,
# - `radius_wire` ⟷ $r_w$.
#
# We also define a nested function for the calculation of $a_l$. For the
# final calculation of the efficiencies, the summation over the different
# orders of the Bessel functions is truncated at $\nu=50$.

# +

def spherical_hn(nu, alpha, derivative=False):

    if derivative:
        return spherical_jn(
            nu, alpha, derivative) + 1j * spherical_yn(
            nu, alpha, derivative)
    else:
        return spherical_jn(nu, alpha) + 1j * spherical_yn(nu, alpha)


def compute_coefficients(nu: int, m: complex, alpha: float) -> float:

    psi = alpha * spherical_jn(nu, alpha)
    dpsi = spherical_jn(
        nu, alpha) + alpha * spherical_jn(nu, alpha, derivative=True)
    psim = alpha * spherical_jn(nu, m * alpha)
    dpsim = spherical_jn(
        nu, m * alpha) + m * alpha * spherical_jn(nu, m * alpha, derivative=True)

    eta = alpha * spherical_hn(nu, alpha)
    deta = spherical_hn(nu, alpha) + alpha * spherical_hn(nu,
                                                          alpha, derivative=True)

    a = (m * psim * dpsi - psi * dpsim) / (m * psim * deta - eta * dpsim)

    b = (psim * dpsi - m * psi * dpsim) / (psim * deta - m * eta * dpsim)

    return a, b


def calculate_analytical_efficiencies(eps: complex, n_bkg: float,
                                      wl0: float, radius_sph: float,
                                      num_n: int = 50) -> Tuple[float, float, float]:

    m = np.sqrt(np.conj(eps)) / n_bkg
    k = 2 * np.pi * n_bkg / wl0

    c = 2 / (k**2 * radius_sph**2)
    alpha = k * radius_sph

    q_ext = 0
    q_sca = 0

    for nu in range(1, num_n + 1):

        a_nu, b_nu = compute_coefficients(nu, m, alpha)
        q_ext += (2 * nu + 1) * c * np.real(a_nu + b_nu)
        q_sca += (2 * nu + 1) * c * (np.abs(a_nu)**2 + np.abs(b_nu)**2)

    return q_ext - q_sca, q_sca, q_ext


eps_au = -1.0782 + 1j * 5.8089
radius_sph = 25
wl0 = 400
n_bkg = 1.0

q_ext_analyt, q_sca_analyt, q_abs_analyt = scattnlay(np.array(
    [2 * np.pi * radius_sph / wl0 * n_bkg],
    dtype=np.complex128),
    np.array(
    [np.sqrt(eps_au) / n_bkg],
    dtype=np.complex128))[1:4]

q_abs_my, q_sca_my, q_ext_my = calculate_analytical_efficiencies(
    eps_au, n_bkg, wl0, radius_sph)

err_abs = np.abs(q_abs_analyt - q_abs_my) / q_abs_analyt
err_sca = np.abs(q_sca_analyt - q_sca_my) / q_sca_analyt
err_ext = np.abs(q_ext_analyt - q_ext_my) / q_ext_analyt

print()
print(f"The analytical absorption efficiency is {q_abs_analyt}")
print(f"The numerical absorption efficiency is {q_abs_my}")
print(f"The numerical absorption efficiency is {q_abs_my/q_abs_analyt}")
print(f"The error is {err_abs*100}%")
print()
print(f"The analytical scattering efficiency is {q_sca_analyt}")
print(f"The numerical scattering efficiency is {q_sca_my}")
print(f"The numerical absorption efficiency is {q_sca_my/q_sca_analyt}")
print(f"The error is {err_sca*100}%")
print()
print(f"The analytical extinction efficiency is {q_ext_analyt}")
print(f"The numerical extinction efficiency is {q_ext_my}")
print(f"The ratio is {q_ext_my/q_ext_analyt}")
print(f"The error is {err_ext*100}%")

assert err_abs < 0.01
assert err_sca < 0.01
assert err_ext < 0.01
