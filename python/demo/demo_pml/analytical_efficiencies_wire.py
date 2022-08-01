# # Calculation of analytical efficiencies
#
# Copyright (C) 2022 Michele Castriotta, Igor Baratta, Jørgen S. Dokken
#
# This file contains a function for the calculation of the absorption,
# scattering and extinction efficiencies of a wire being
# hit normally by a TM-polarized electromagnetic wave. The formula
# are taken from:
#
# Milton Kerker, "The Scattering of Light and Other Electromagnetic
# Radiation", Chapter 6, Elsevier, 1969.
#
# ## Implementation
#
# First of all, let's define the parameters of the problem:
#
# - $n = \sqrt{\varepsilon}$: refractive index of the wire,
# - $n_b$: refractive index of the background medium,
# - $m = n/n_b$: relative refractive index of the wire,
# - $\lambda_0$: wavelength of the electromagnetic wave,
# - $r_w$: radius of the cross-section of the wire,
# - $\alpha = 2\pi r_w n_b/\lambda_0$.
#
# Now, let's define the $a_l$ coefficients as:
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
#
# - $J_\nu(x)$: $\nu$-th order Bessel function of the first kind,
# - $J_\nu^{\prime}(x)$: first derivative with respect to $x$ of the
# $\nu$-th order Bessel function of the first kind,
# - $H_\nu^{(2)}(x)$: $\nu$-th order Hankel function of the second kind,
# - $H_\nu^{(2){\prime}}(x)$: first derivative with respect to $x$ of the
# $\nu$-th order Hankel function of the second kind.
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
#
#
# The modules that will be used are imported:

import numpy as np
from scipy.special import h2vp, hankel2, jv, jvp

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


def calculate_analytical_efficiencies(eps, n_bkg, wl0, radius_wire):

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

    m = np.sqrt(np.conj(eps)) / n_bkg

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
