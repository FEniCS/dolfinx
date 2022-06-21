# # Calculation of analytical efficiencies
# This file contains a function for the calculation of the absorption,
# scattering and extinction efficiencies of a wire being 
# hit normally by a TM-polarized electromagnetic wave. The formula 
# are taken from:
#
# Milton Kerker, "The Scattering of Light and Other Electromagnetic 
# Radiation", Chapter 6, Elsevier, 1969.
# 
## Equations
# First of all, let's define the parameters of the problem:
#
# - $n = \sqrt{\varepsilon}$: refractive index of the wire,
# - $n_b$: refractive index of the background medium,
# - $m = n/n_b$: relative refractive index of the wire,
# - $\lambda_0$: wavelength of the electromagnetic wave,
# - $r$: radius of the cross-section of the wire,
# - $\alpha = 2\pi r n_b/\lambda_0$.
#
# Now, let's define the scattering coefficients $a_l$ as:
# 
# $$
# \begin{equation}
# a_l=\frac{J_l(\alpha) J_l^{\prime}(m \alpha)-m J_l(m \alpha) 
# J_l^{\prime}(\alpha)}{H_l^{(2)}(\alpha) J_l^{\prime}(m \alpha)
# -m J_l(m \alpha) H_l^{(2){\prime}}(\alpha)}
# \end{equation}
# $$
# 
# where:
# 
# - $J_l(x)$: $l$-th order Bessel function of the first kind,
# - $J_l^{\prime}(x)$: first derivative with respect to $x$ of the
# $l$-th order Bessel function of the first kind,
# - $H_l^{(2)}(x)$: $l$-th order Hankel function of the second kind,
# - $H_l^{(2){\prime}}(x)$: first derivative with respect to $x$ of the 
# $l$-th order Hankel function of the second kind.
#
# We can now calculate the scattering, extinction and absorption
# efficiencies as:
#
# $$
# \begin{align}
# & q_{\mathrm{sca}}=(2 / \alpha)\left[\left|a_0\right|^{2}
# +2 \sum_{l=1}^{\infty}\left|a_l\right|^{2}\right] \\
# & q_{\mathrm{ext}}=(2 / \alpha) \operatorname{Re}\left[ a_0
# +2 \sum_{l=1}^{\infty} a_l\right] \\
# & q_{\mathrm{abs}} = q_{\mathrm{ext}} - q_{\mathrm{sca}}
# \end{align} 
# $$
#
# ## Implementation
# The modules that will be used are imported:

# +
import numpy as np
from scipy.special import h2vp, hankel2, jv, jvp
# -

# The functions that we import from `scipy.special` correspond to:
#
# - `jv(l, x)` ⟷ $J_l(x)$,
# - `jvp(l, x, 1)` ⟷ $J_l^{\prime}(x)$,
# - `hankel2(l, x)` ⟷ $H_l^{(2)}(x)$,
# - `h2vp(l, x, 1)` ⟷ $H_l^{(2){\prime}}(x)$.
# 
# Next, we define a function for calculating the analytical efficiencies
# in Python. The inputs of the function are:
#
# - `reps` ⟷ $\operatorname{Re}(\varepsilon)$,
# - `ieps` ⟷ $\operatorname{Im}(\varepsilon)$,
# - `n_bkg` ⟷ $n_b$,
# - `wl0` ⟷ $\lambda_0$,
# - `radius_wire` ⟷ $r$.
#
# We also define a nested function for the calculation of $a_l$. For the
# final calculation of the efficiencies, the summation over the different
# orders of the Bessel functions is truncated at $l=50$.


# +
def calculate_analytical_efficiencies(reps, ieps, n_bkg, wl0, radius_wire):

    def a_coeff(l, m, alpha):

        J_l_alpha = jv(l, alpha)
        J_l_malpha = jv(l, m * alpha)
        J_l_alpha_p = jvp(l, alpha, 1)
        J_l_malpha_p = jvp(l, m * alpha, 1)

        H_l_alpha = hankel2(l, alpha)
        H_l_alpha_p = h2vp(l, alpha, 1)

        al_num = J_l_alpha * J_l_malpha_p - m * J_l_malpha * J_l_alpha_p
        al_den = H_l_alpha * J_l_malpha_p - m * J_l_malpha * H_l_alpha_p

        al = al_num / al_den

        return al

    m = np.sqrt(reps - 1j * ieps) / n_bkg

    alpha = 2 * np.pi * radius_wire / wl0 * n_bkg
    c = 2 / alpha

    num_n = 50

    for l in range(0, num_n + 1):

        if l == 0:

            q_ext = c * np.real(a_coeff(l, m, alpha))
            q_sca = c * np.abs(a_coeff(l, m, alpha))**2

        else:

            q_ext += c * 2 * np.real(a_coeff(l, m, alpha))
            q_sca += c * 2 * np.abs(a_coeff(l, m, alpha))**2

    q_abs = q_ext - q_sca

    return q_abs, q_sca, q_ext
# -