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
# The analytical solutions for the half-loaded waveguide with perfect
# electric conducting walls are described in Harrington's *Time-harmonic
# electromagnetic fields*. We will skip the full derivation, and we just
# mention that the problem can be decoupled into $\mathrm{TE}_x$ and
# $\mathrm{TM}_x$ modes, and the possible $k_z$ can be found by solving
# a set of transcendental equations, which is shown here below:
#
#
# $$
# \textrm{For TE}_x \textrm{ modes}:
# \begin{cases}
# &k_{x d}^{2}+\left(\frac{n \pi}{w}\right)^{2}+k_{z}^{2}=k_0^{2}
# \varepsilon_{d} \\
# &k_{x v}^{2}+\left(\frac{n \pi}{w}\right)^{2}+k_{z}^{2}=k_0^{2}
# \varepsilon_{v} \\
# & k_{x d} \cot k_{x d} d + k_{x v} \cot \left[k_{x v}(h-d)\right] = 0
# \end{cases}
# $$
#
# $$
# \textrm{For TM}_x \textrm{ modes}:
# \begin{cases}
# &k_{x d}^{2}+\left(\frac{n \pi}{w}\right)^{2}+k_{z}^{2}=
# k_0^{2} \varepsilon_{d} \\
# &k_{x v}^{2}+\left(\frac{n \pi}{w}\right)^{2}+k_{z}^{2}=
# k_0^{2} \varepsilon_{v} \\
# & \frac{k_{x d}}{\varepsilon_{d}} \tan k_{x d} d +
# \frac{k_{x v}}{\varepsilon_{v}} \tan \left[k_{x v}(h-d)\right] = 0
# \end{cases}
# $$
#
# with:
# - $\varepsilon_d\rightarrow$ dielectric permittivity
# - $\varepsilon_v\rightarrow$ vacuum permittivity
# - $w\rightarrow$ total width of the waveguide
# - $h\rightarrow$ total height of the waveguide
# - $d\rightarrow$ height of the dielectric fraction
# - $k_0\rightarrow$ vacuum wavevector
# - $k_{xd}\rightarrow$ $x$ component of the wavevector in the dielectric
# - $k_{xv}\rightarrow$ $x$ component of the wavevector in the vacuum
# - $\frac{n \pi}{w} = k_y\rightarrow$ $y$ component of the wavevector
#   for different $n$ harmonic numbers (we assume $n=1$ for the sake of
#   simplicity)
#
# Let's define the set of equations with the $\tan$ and $\cot$ function:

import numpy as np


def TMx_condition(
    kx_d: complex, kx_v: complex, eps_d: complex, eps_v: complex, d: float, h: float
) -> float:
    return kx_d / eps_d * np.tan(kx_d * d) + kx_v / eps_v * np.tan(kx_v * (h - d))


def TEx_condition(kx_d: complex, kx_v: complex, d: float, h: float) -> float:
    return kx_d / np.tan(kx_d * d) + kx_v / np.tan(kx_v * (h - d))


# Then, we can define the `verify_mode` function, to check whether a
# certain $k_z$ satisfy the equations (below a certain threshold). In
# other words, we provide a certain $k_z$, together with the geometrical
# and optical parameters of the waveguide, and `verify_mode()` checks
# whether the last equations for the $\mathrm{TE}_x$ or $\mathrm{TM}_x$
# modes are close to $0$.


def verify_mode(
    kz: complex,
    w: float,
    h: float,
    d: float,
    lmbd0: float,
    eps_d: complex,
    eps_v: complex,
    threshold: float,
) -> np.bool_:
    k0 = 2 * np.pi / lmbd0
    ky = np.pi / w  # we assume n = 1
    kx_d_target = np.sqrt(k0**2 * eps_d - ky**2 + -(kz**2) + 0j)
    alpha = kx_d_target**2
    beta = alpha - k0**2 * (eps_d - eps_v)
    kx_v = np.sqrt(beta)
    kx_d = np.sqrt(alpha)
    f_tm = TMx_condition(kx_d, kx_v, eps_d, eps_v, d, h)
    f_te = TEx_condition(kx_d, kx_v, d, h)
    return np.isclose(f_tm, 0, atol=threshold) or np.isclose(f_te, 0, atol=threshold)
