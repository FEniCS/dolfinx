# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Analytical solutions for the half-loaded waveguide
#
# The analytical solutions for the half-loaded waveguide
# with perfect electric conducting walls are
# described in Harrington's *Time-harmonic electromagnetic fields*.
# We will skip the full derivation, and we just mention that
# the problem can be decoupled into $\mathrm{TE}_x$ and
# $\mathrm{TM}_x$ modes, and the possible $k_z$ can be found by
# solving a set of transcendental equations, which is shown here below:
#
#
# $$
# \begin{aligned}
# \textrm{For TE}_x \textrm{ modes}:
# \begin{cases}
# &k_{x d}^{2}+\left(\frac{n \pi}{b}\right)^{2}+k_{z}^{2}=k_0^{2}
# \varepsilon_{d} \\
# &k_{x v}^{2}+\left(\frac{n \pi}{b}\right)^{2}+k_{z}^{2}=k_0^{2}
# \varepsilon_{v} \\
# & k_{x d} \cot k_{x d} d + k_{x v} \cot \left[k_{x v}(h-d)\right] = 0
# \end{cases}
# \end{aligned}
# $$
#
# $$
# \begin{aligned}
# \textrm{For TM}_x \textrm{ modes}:
# \begin{cases}
# &k_{x d}^{2}+\left(\frac{n \pi}{b}\right)^{2}+k_{z}^{2}=
# k_0^{2} \varepsilon_{d} \\
# &k_{x v}^{2}+\left(\frac{n \pi}{b}\right)^{2}+k_{z}^{2}=
# k_0^{2} \varepsilon_{v} \\
# & \frac{k_{x d}}{\varepsilon_{d}} \tan k_{x d} d +
# \frac{k_{x v}}{\varepsilon_{v}} \tan \left[k_{x v}(h-d)\right] = 0
# \end{cases}
# \end{aligned}
# $$
#
# with:
# - $\varepsilon_d\rightarrow$ dielectric permittivity
# - $\varepsilon_v\rightarrow$ vacuum permittivity
# - $h\rightarrow$ height of the waveguide
# - $d\rightarrow$ height of the dielectric
# - $k_0\rightarrow$ vacuum wavevector
# - $k_{xd}\rightarrow$ $x$ component of the wavevector in the dielectric
# - $k_{xv}\rightarrow$ $x$ component of the wavevector in the vacuum
# - $\frac{n \pi}{b} = k_y\rightarrow$ $y$ component of the wavevector for
# different $n$ harmonic numbers
#
# Let's define the set of equations with the $\tan$ and $\cot$ function:

import numpy as np


def TMx_condition(kx_d, kx_v, eps_d, eps_v, d, h):
    return (kx_d / eps_d * np.tan(kx_d * d)
            + kx_v / eps_v * np.tan(kx_v * (h - d)))


def TEx_condition(kx_d, kx_v, d, h):
    return kx_d / np.tan(kx_d * d) + kx_v / np.tan(kx_v * (h - d))

# Then, we can define the `verify_mode` function, to check whether a certain
# $k_z$ satisfy the equations (below a certain threshold):


def verify_mode(
        kz, w, h, d, lmbd0, eps_d, eps_v, threshold):

    k0 = 2 * np.pi / lmbd0

    n = 1
    ky = n * np.pi / w

    kx_d_target = np.sqrt(k0**2 * eps_d - ky**2 + - kz**2 + 0j)

    alpha = kx_d_target**2

    beta = alpha - k0**2 * (eps_d - eps_v)

    kx_v = np.sqrt(beta)
    kx_d = np.sqrt(alpha)

    f_tm = TMx_condition(kx_d, kx_v, eps_d, eps_v, d, h)
    f_te = TEx_condition(kx_d, kx_v, d, h)

    return np.isclose(
        f_tm, 0, atol=threshold) or np.isclose(
        f_te, 0, atol=threshold)
