import numpy as np
from scipy.optimize import fsolve, root, show_options
from functools import partial
import pytest

def TMx_solver(k0, eps1, eps2, h, d, factor, method):

    def transcendental_equation(kx, k0, eps1, eps2, h, d):
        kx1, kx2 = kx
        f1 = kx1**2 - kx2**2 - k0**2 * (eps1 - eps2)
        f2 = kx1 / eps1 * np.tan(kx1 * d) + kx2 / eps2 * np.tan(kx2 * (h - d))
        return [f1, f2]

    partial_te = partial(transcendental_equation, k0=k0, eps1=eps1,
                         eps2=eps2, h=h, d=d)

    return root(partial_te, x0=(factor*k0, )*2, 
                            method = method, 
                            options = {"maxiter": 1000000,
                                       "ftol": 1e-6})

#@pytest.mark.parametrize("method", ['lm', 'broyden1', 'broyden2', 
#                                    'anderson', 'linearmixing',
#                                    'diagbroyden', 'excitingmixing', 'krylov'])
#
#def test_solver(method):
#    
#    MHz = 10**6
#    f0 = 200 * MHz
#    c0 = 3*10**8 # m/s
#    lmbd0 = c0 / f0 # m
#    k0 = 2 * np.pi / lmbd0 # m^-1
#
#    eps1 = 2.45
#    eps2 = 1
#    l = 1
#    h = 0.45 * l
#    d = 0.5 * h
#    sols = TMx_solver(k0, eps1, eps2, h, d, method)
#
#    print()
#    print(sols)
#    print()

#@pytest.mark.parametrize("method", ['diagbroyden', 'krylov'])

MHz = 10**6
f0 = 200 * MHz
c0 = 3*10**8 # m/s
lmbd0 = c0 / f0 # m
k0 = 2 * np.pi / lmbd0 # m^-1

@pytest.mark.parametrize("k0", [k0])
@pytest.mark.parametrize("factor", np.linspace(0.01, 1, 100))

def test_solver(k0, factor):
    
    eps1 = 2.45
    eps2 = 1
    l = 1
    h = 0.45 * l
    d = 0.5 * h
    sols = TMx_solver(k0, eps1, eps2, h, d, factor, 'diagbroyden')

    kx1 = sols.x[0]
    kx2 = sols.x[1]

    is_success = sols.success
    kz_1 = np.sqrt(k0**2*eps1 - kx1**2 + 0j)
    kz_2 = np.sqrt(k0**2*eps2 - kx2**2 + 0j)
    is_kz_real_1 = np.isclose(kz_1.imag, 0)
    is_kz_real_2 = np.isclose(kz_2.imag, 0)

    assert is_success
    assert is_kz_real_1
    assert is_kz_real_2

    if is_success:
        print()
        print((kx1**2 - kx2**2)/k0**2 - (eps1 - eps2))
        print(sols)
        print()

    if (is_success and is_kz_real_1 and is_kz_real_2):
        print()
        print(f"factor: {factor}")
        print(f"kz_1: {np.sqrt(k0**2*eps1 - kx1**2)}")
        print(f"kz_2: {np.sqrt(k0**2*eps2 - kx2**2)}")
        print(sols)
        print()