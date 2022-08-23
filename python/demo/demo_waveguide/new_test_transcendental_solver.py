from re import A
import numpy as np
from scipy.optimize import fsolve, root, show_options
from functools import partial
import pytest

eps1 = 2.45
eps2 = 1
l = 1
h = 0.45 * l
d = 0.5 * h

lmbd0 = h/0.2
k0 = 2 * np.pi / lmbd0

ndim = 1000000
beta = np.linspace(-k0**2*eps2, k0**2*eps2, ndim, 
                    endpoint=False, dtype=np.complex128)
alpha = k0**2*(eps1 - eps2) + beta

kx2 = np.sqrt(beta)
kx1 = np.sqrt(alpha)

def TMx_condition(kx1, kx2, eps1, eps2, d, h):
    return kx1 / eps1 * np.tan(kx1 * d) + kx2 / eps2 * np.tan(kx2 * (h - d))

def TEx_condition(kx1, kx2, d, h):
    return kx1 / np.tan(kx1 * d) + kx2 / np.tan(kx2 * (h - d))

f_tm = TMx_condition(kx1, kx2, eps1, eps2, d, h)
f_te = TEx_condition(kx1, kx2, d, h)

kx1_tm = np.extract(np.isclose(f_tm, np.zeros(ndim), atol=1e-4), kx1)
kx2_tm = np.extract(np.isclose(f_tm, np.zeros(ndim), atol=1e-4), kx2)

kx1_te = np.extract(np.isclose(f_te, np.zeros(ndim), atol=1e-4), kx1)
kx2_te = np.extract(np.isclose(f_te, np.zeros(ndim), atol=1e-4), kx2)

print(kx1_tm)
print(kx2_tm)

print(kx1_te)
print(kx2_te)

kx1_tm_mean = np.mean(kx1_tm)
kx2_tm_mean = np.mean(kx2_tm)
kx1_te_mean = np.mean(kx1_te)
kx2_te_mean = np.mean(kx2_te)

n = 1
ky = n*np.pi/l

kz_tm = np.sqrt(k0**2*eps1 - ky**2 - kx1_tm_mean**2 + 0j).real/k0
kz_te = np.sqrt(k0**2*eps1 - ky**2 - kx1_te_mean**2 + 0j).real/k0

print(kz_tm)
print(kz_te)