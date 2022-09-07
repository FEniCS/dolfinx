import numpy as np


def TMx_condition(kx_d, kx_v, eps_d, eps_v, d, h):
    return kx_d / eps_d * np.tan(kx_d * d) + kx_v / eps_v * np.tan(kx_v * (h - d))


def TEx_condition(kx_d, kx_v, d, h):
    return kx_d / np.tan(kx_d * d) + kx_v / np.tan(kx_v * (h - d))


def verify_mode(
        kz, l, h, d, lmbd0, eps_d, eps_v, threshold):

    k0 = 2 * np.pi / lmbd0

    n = 1
    ky = n * np.pi / l

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
