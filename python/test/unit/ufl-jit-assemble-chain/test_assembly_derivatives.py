"""System integration tests for ufl-derivative-jit-assembly chain."""

# Copyright (C) 2011 Martin S. Alnaes
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import math

import numpy
import pytest
from dolfinx import RectangleMesh, UnitIntervalMesh
from mpi4py import MPI
from ufl import (FacetNormal, SpatialCoordinate, acos, as_matrix, as_vector,
                 asin, atan, cos, cross, det, dev, diff, div, dot, ds, dx,
                 elem_div, elem_mult, elem_op, elem_pow, erf, exp, grad, inner,
                 ln, outer, sin, skew, sym, tan, tr)


@pytest.mark.skip
def test_diff_then_integrate():

    # Define 1D geometry
    n = 21
    mesh = UnitIntervalMesh(MPI.COMM_WORLD, n)

    # Shift and scale mesh
    x0, x1 = 1.5, 3.14
    mesh.coordinates()[:] *= (x1 - x0)
    mesh.coordinates()[:] += x0

    x = SpatialCoordinate(mesh)[0]
    xs = 0.1 + 0.8 * x / x1  # scaled to be within [0.1,0.9]

    # Define list of expressions to test, and configure
    # accuracies these expressions are known to pass with.
    # The reason some functions are less accurately integrated is
    # likely that the default choice of quadrature rule is not perfect
    F_list = []

    def reg(exprs, acc=10):
        for expr in exprs:
            F_list.append((expr, acc))

    # FIXME: 0*dx and 1*dx fails in the ufl-ffcx-jit framework somewhere
    # reg([Constant(0.0, cell=cell)])
    # reg([Constant(1.0, cell=cell)])
    monomial_list = [x**q for q in range(2, 6)]
    reg(monomial_list)
    reg([2.3 * p + 4.5 * q for p in monomial_list for q in monomial_list])
    reg([x**x])
    reg([x**(x**2)], 8)
    reg([x**(x**3)], 6)
    reg([x**(x**4)], 2)
    # Special functions:
    reg([atan(xs)], 8)
    reg([sin(x), cos(x), exp(x)], 5)
    reg([ln(xs), pow(x, 2.7), pow(2.7, x)], 3)
    reg([asin(xs), acos(xs)], 1)
    reg([tan(xs)], 7)

    try:
        import scipy
    except ImportError:
        scipy = None

    if hasattr(math, 'erf') or scipy is not None:
        reg([erf(xs)])
    else:
        print("Warning: skipping test of erf, old python version and no scipy.")

    # if 0:
    #     print("Warning: skipping tests of bessel functions, doesn't build on all platforms.")
    # elif scipy is None:
    #     print("Warning: skipping tests of bessel functions, missing scipy.")
    # else:
    #     for nu in (0, 1, 2):
    #         # Many of these are possibly more accurately integrated,
    #         # but 4 covers all and is sufficient for this test
    #         reg([bessel_J(nu, xs), bessel_Y(nu, xs), bessel_I(nu, xs), bessel_K(nu, xs)], 4)

    # To handle tensor algebra, make an x dependent input tensor
    # xx and square all expressions
    def reg2(exprs, acc=10):
        for expr in exprs:
            F_list.append((inner(expr, expr), acc))
    xx = as_matrix([[2 * x**2, 3 * x**3], [11 * x**5, 7 * x**4]])
    x3v = as_vector([3 * x**2, 5 * x**3, 7 * x**4])
    cc = as_matrix([[2, 3], [4, 5]])
    reg2([xx])
    reg2([x3v])
    reg2([cross(3 * x3v, as_vector([-x3v[1], x3v[0], x3v[2]]))])
    reg2([xx.T])
    reg2([tr(xx)])
    reg2([det(xx)])
    reg2([dot(xx, 0.1 * xx)])
    reg2([outer(xx, xx.T)])
    reg2([dev(xx)])
    reg2([sym(xx)])
    reg2([skew(xx)])
    reg2([elem_mult(7 * xx, cc)])
    reg2([elem_div(7 * xx, xx + cc)])
    reg2([elem_pow(1e-3 * xx, 1e-3 * cc)])
    reg2([elem_pow(1e-3 * cc, 1e-3 * xx)])
    reg2([elem_op(lambda z: sin(z) + 2, 0.03 * xx)], 2)  # pretty inaccurate...

    # FIXME: Add tests for all UFL operators:
    # These cause discontinuities and may be harder to test in the
    # above fashion:
    # 'inv', 'cofac',
    # 'eq', 'ne', 'le', 'ge', 'lt', 'gt', 'And', 'Or', 'Not',
    # 'conditional', 'sign',
    # 'jump', 'avg',
    # 'LiftingFunction', 'LiftingOperator',

    # FIXME: Test other derivatives: (but algorithms for operator
    # derivatives are the same!):
    # 'variable', 'diff',
    # 'Dx', 'grad', 'div', 'curl', 'rot', 'Dn', 'exterior_derivative',

    # Run through all operators defined above and compare integrals
    debug = 0
    for F, acc in F_list:
        # Apply UFL differentiation
        f = diff(F, SpatialCoordinate(mesh))[..., 0]
        if debug:
            print(F)
            print(x)
            print(f)

        # Apply integration with DOLFINX
        # (also passes through form compilation and jit)
        M = f * dx
        f_integral = assemble_scalar(M)  # noqa
        f_integral = mesh.mpi_comm().allreduce(f_integral, op=MPI.SUM)

        # Compute integral of f manually from anti-derivative F
        # (passes through pybind11 interface and uses UFL evaluation)
        F_diff = F((x1,)) - F((x0,))

        # Compare results. Using custom relative delta instead
        # of decimal digits here because some numbers are >> 1.
        delta = min(abs(f_integral), abs(F_diff)) * 10**-acc
        assert f_integral - F_diff <= delta


@pytest.mark.skip
def test_div_grad_then_integrate_over_cells_and_boundary():

    # Define 2D geometry
    n = 10
    mesh = RectangleMesh([numpy.array([0.0, 0.0, 0.0]),
                          numpy.array([2.0, 3.0, 0.0])], 2 * n, 3 * n)

    x, y = SpatialCoordinate(mesh)
    xs = 0.1 + 0.8 * x / 2  # scaled to be within [0.1,0.9]
#    ys = 0.1 + 0.8 * y / 3  # scaled to be within [0.1,0.9]
    n = FacetNormal(mesh)

    # Define list of expressions to test, and configure accuracies
    # these expressions are known to pass with.  The reason some
    # functions are less accurately integrated is likely that the
    # default choice of quadrature rule is not perfect
    F_list = []

    def reg(exprs, acc=10):
        for expr in exprs:
            F_list.append((expr, acc))

    # FIXME: 0*dx and 1*dx fails in the ufl-ffcx-jit framework somewhere
    # reg([Constant(0.0, cell=cell)])
    # reg([Constant(1.0, cell=cell)])
    monomial_list = [x**q for q in range(2, 6)]
    reg(monomial_list)
    reg([2.3 * p + 4.5 * q for p in monomial_list for q in monomial_list])
    reg([xs**xs])
    reg([xs**(xs**2)], 8)  # Note: Accuracies here are from 1D case, not checked against 2D results.
    reg([xs**(xs**3)], 6)
    reg([xs**(xs**4)], 2)
    # Special functions:
    reg([atan(xs)], 8)
    reg([sin(x), cos(x), exp(x)], 5)
    reg([ln(xs), pow(x, 2.7), pow(2.7, x)], 3)
    reg([asin(xs), acos(xs)], 1)
    reg([tan(xs)], 7)

    # To handle tensor algebra, make an x dependent input tensor
    # xx and square all expressions
    def reg2(exprs, acc=10):
        for expr in exprs:
            F_list.append((inner(expr, expr), acc))
    xx = as_matrix([[2 * x**2, 3 * x**3], [11 * x**5, 7 * x**4]])
    xxs = as_matrix([[2 * xs**2, 3 * xs**3], [11 * xs**5, 7 * xs**4]])
    x3v = as_vector([3 * x**2, 5 * x**3, 7 * x**4])
    cc = as_matrix([[2, 3], [4, 5]])
    reg2([xx])  # TODO: Make unit test for UFL from this, results in listtensor with free indices
    reg2([x3v])
    reg2([cross(3 * x3v, as_vector([-x3v[1], x3v[0], x3v[2]]))])
    reg2([xx.T])
    reg2([tr(xx)])
    reg2([det(xx)])
    reg2([dot(xx, 0.1 * xx)])
    reg2([outer(xx, xx.T)])
    reg2([dev(xx)])
    reg2([sym(xx)])
    reg2([skew(xx)])
    reg2([elem_mult(7 * xx, cc)])
    reg2([elem_div(7 * xx, xx + cc)])
    reg2([elem_pow(1e-3 * xxs, 1e-3 * cc)])
    reg2([elem_pow(1e-3 * cc, 1e-3 * xx)])
    reg2([elem_op(lambda z: sin(z) + 2, 0.03 * xx)], 2)  # pretty inaccurate...

    # FIXME: Add tests for all UFL operators:
    # These cause discontinuities and may be harder to test in the
    # above fashion:
    # 'inv', 'cofac',
    # 'eq', 'ne', 'le', 'ge', 'lt', 'gt', 'And', 'Or', 'Not',
    # 'conditional', 'sign',
    # 'jump', 'avg',
    # 'LiftingFunction', 'LiftingOperator',

    # FIXME: Test other derivatives: (but algorithms for operator
    # derivatives are the same!):
    # 'variable', 'diff',
    # 'Dx', 'grad', 'div', 'curl', 'rot', 'Dn', 'exterior_derivative',

    # Run through all operators defined above and compare integrals
    debug = 0
    if debug:
        F_list = F_list[1:]

    for F, acc in F_list:
        if debug:
            print('\n', "F:", str(F))

        # Integrate over domain and its boundary
        int_dx = assemble(div(grad(F)) * dx(mesh))  # noqa
        int_ds = assemble(dot(grad(F), n) * ds(mesh))  # noqa

        if debug:
            print(int_dx, int_ds)

        # Compare results. Using custom relative delta instead of
        # decimal digits here because some numbers are >> 1.
        delta = min(abs(int_dx), abs(int_ds)) * 10**-acc
        assert int_dx - int_ds <= delta
