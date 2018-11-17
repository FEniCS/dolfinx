# Copyright (C) 2011-2014 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the Function class"""

import math

import numba
import numpy
import pytest

import ufl
from dolfin import (DOLFIN_EPS, MPI, Expression, Function, FunctionSpace,
                    Point, TensorFunctionSpace, UnitCubeMesh,
                    VectorFunctionSpace, Vertex, cpp, function, interpolate,
                    lt)
from dolfin_utils.test.fixtures import fixture
from dolfin_utils.test.skips import skip_in_parallel


@fixture
def mesh():
    return UnitCubeMesh(MPI.comm_world, 3, 3, 3)


@fixture
def R(mesh):
    return FunctionSpace(mesh, ('R', 0))


@fixture
def V(mesh):
    return FunctionSpace(mesh, ('CG', 1))


@fixture
def W(mesh):
    return VectorFunctionSpace(mesh, ('CG', 1))


@fixture
def Q(mesh):
    return TensorFunctionSpace(mesh, ('CG', 1))


def test_name_argument(W):
    u = Function(W)
    v = Function(W, name="v")
    assert u.name() == "f_%d" % u.count()
    assert v.name() == "v"
    assert str(v) == "v"


def test_compute_point_values(V, W, mesh):
    from numpy import all
    u = Function(V)
    v = Function(W)

    u.vector()[:] = 1.
    v.vector()[:] = 1.

    u.vector().update_ghosts()
    v.vector().update_ghosts()

    u_values = u.compute_point_values(mesh)
    v_values = v.compute_point_values(mesh)

    u_ones = numpy.ones_like(u_values, dtype=numpy.float64)
    assert all(numpy.isclose(u_values, u_ones))

    v_ones = numpy.ones_like(v_values, dtype=numpy.float64)
    assert all(numpy.isclose(v_values, v_ones))

    u_values2 = u.compute_point_values()

    assert all(u_values == u_values2)


@pytest.mark.skip
def test_assign(V, W):
    for V0, V1, vector_space in [(V, W, False), (W, V, True)]:
        u = Function(V0)
        u0 = Function(V0)
        u1 = Function(V0)
        u2 = Function(V0)
        u3 = Function(V1)

        u.vector()[:] = 1.0
        u0.vector()[:] = 2.0
        u1.vector()[:] = 3.0
        u2.vector()[:] = 4.0
        u3.vector()[:] = 5.0

        uu = Function(V0)
        uu.assign(2 * u)
        assert uu.vector().get_local().sum() == u0.vector().get_local().sum()

        uu = Function(V1)
        uu.assign(3 * u)
        assert uu.vector().get_local().sum() == u1.vector().get_local().sum()

        # Test complex assignment
        expr = 3 * u - 4 * u1 - 0.1 * 4 * u * 4 + u2 + 3 * u0 / 3. / 0.5
        expr_scalar = 3 - 4 * 3 - 0.1 * 4 * 4 + 4. + 3 * 2. / 3. / 0.5
        uu.assign(expr)
        assert (round(
            uu.vector().get_local().sum() - float(
                expr_scalar * uu.vector().size()), 7) == 0)

        # Test expression scaling
        expr = 3 * expr
        expr_scalar *= 3
        uu.assign(expr)
        assert (round(
            uu.vector().get_local().sum() - float(
                expr_scalar * uu.vector().size()), 7) == 0)

        # Test expression scaling
        expr = expr / 4.5
        expr_scalar /= 4.5
        uu.assign(expr)
        assert (round(
            uu.vector().get_local().sum() - float(
                expr_scalar * uu.vector().size()), 7) == 0)

        # Test self assignment
        expr = 3 * u - 5.0 * u2 + u1 - 5 * u
        expr_scalar = 3 - 5 * 4. + 3. - 5
        u.assign(expr)
        assert (round(
            u.vector().get_local().sum() - float(
                expr_scalar * u.vector().size()), 7) == 0)

        # Test zero assignment
        u.assign(-u2 / 2 + 2 * u1 - u1 / 0.5 + u2 * 0.5)
        assert round(u.vector().get_local().sum() - 0.0, 7) == 0

        # Test erroneous assignments
        uu = Function(V1)

        @function.expression.numba_eval
        def expr_eval(values, x, cell_idx):
            values[:, 0] = 1.0

        f = Expression(expr_eval)

        with pytest.raises(RuntimeError):
            uu.assign(1.0)
        with pytest.raises(RuntimeError):
            uu.assign(4 * f)

        if not vector_space:
            with pytest.raises(RuntimeError):
                uu.assign(u * u0)
            with pytest.raises(RuntimeError):
                uu.assign(4 / u0)
            with pytest.raises(RuntimeError):
                uu.assign(4 * u * u1)


def test_call(R, V, W, Q, mesh):
    u0 = Function(R)
    u1 = Function(V)
    u2 = Function(W)
    u3 = Function(Q)

    @function.expression.numba_eval
    def expr_eval1(values, x, cell_idx):
        values[:, 0] = x[:, 0] + x[:, 1] + x[:, 2]

    e1 = Expression(expr_eval1)

    @function.expression.numba_eval
    def expr_eval2(values, x, cell_idx):
        values[:, 0] = x[:, 0] + x[:, 1] + x[:, 2]
        values[:, 1] = x[:, 0] - x[:, 1] - x[:, 2]
        values[:, 2] = x[:, 0] + x[:, 1] + x[:, 2]

    e2 = Expression(expr_eval2, shape=(3, ))

    @function.expression.numba_eval
    def expr_eval3(values, x, cell_idx):
        values[:, 0] = x[:, 0] + x[:, 1] + x[:, 2]
        values[:, 1] = x[:, 0] - x[:, 1] - x[:, 2]
        values[:, 2] = x[:, 0] + x[:, 1] + x[:, 2]
        values[:, 3] = x[:, 0]
        values[:, 4] = x[:, 1]
        values[:, 5] = x[:, 2]
        values[:, 6] = -x[:, 0]
        values[:, 7] = -x[:, 1]
        values[:, 8] = -x[:, 2]

    e3 = Expression(expr_eval3, shape=(3, 3))

    u0.vector()[:] = 1.0
    u1.interpolate(e1)
    u2.interpolate(e2)
    u3.interpolate(e3)

    p0 = ((Vertex(mesh, 0).point() + Vertex(mesh, 1).point()) / 2.0).array()
    x0 = (mesh.geometry.x(0) + mesh.geometry.x(1)) / 2.0

    assert numpy.allclose(u0(x0), u0(x0))
    assert numpy.allclose(u0(x0), u0(p0))
    assert numpy.allclose(u1(x0), u1(x0))
    assert numpy.allclose(u1(x0), u1(p0))
    assert numpy.allclose(u2(x0)[0], u1(p0))

    assert numpy.allclose(u2(x0), u2(p0))
    assert numpy.allclose(u3(x0)[:3], u2(x0), rtol=1e-15, atol=1e-15)

    p0_list = [p for p in p0]
    x0_list = [x for x in x0]
    assert numpy.allclose(u0(x0_list), u0(x0_list))
    assert numpy.allclose(u0(x0_list), u0(p0_list))

    with pytest.raises(ValueError):
        u0([0, 0, 0, 0])
    with pytest.raises(ValueError):
        u0([0, 0])


def test_scalar_conditions(R):
    c = Function(R)
    c.vector()[:] = 1.5

    # Float conversion does not interfere with boolean ufl expressions
    assert isinstance(lt(c, 3), ufl.classes.LT)
    assert not isinstance(lt(c, 3), bool)

    # Float conversion is not implicit in boolean Python expressions
    assert isinstance(c < 3, ufl.classes.LT)
    assert not isinstance(c < 3, bool)

    # == is used in ufl to compare equivalent representations,
    # <,> result in LT/GT expressions, bool conversion is illegal

    # Note that 1.5 < 0 == False == 1.5 < 1, but that is not what we
    # compare here:
    assert not (c < 0) == (c < 1)
    # This protects from "if c < 0: ..." misuse:
    with pytest.raises(ufl.UFLException):
        bool(c < 0)
    with pytest.raises(ufl.UFLException):
        not c < 0


def test_interpolation_mismatch_rank0(W):
    @function.expression.numba_eval
    def expr_eval(values, x, cell_idx):
        values[:, 0] = 1.0

    f = Expression(expr_eval, shape=())
    with pytest.raises(RuntimeError):
        interpolate(f, W)


def test_interpolation_mismatch_rank1(W):
    @function.expression.numba_eval
    def expr_eval(values, x, cell_idx):
        values[:, 0] = 1.0
        values[:, 1] = 1.0

    f = Expression(expr_eval, shape=(2, ))
    with pytest.raises(RuntimeError):
        interpolate(f, W)


def test_interpolation_rank0(V):
    @function.expression.numba_eval
    def expr_eval(values, x, cell_idx):
        values[:, 0] = 1.0

    f = Expression(expr_eval, shape=())
    w = interpolate(f, V)
    x = w.vector()
    assert MPI.max(MPI.comm_world, abs(x.get_local()).max()) == 1
    assert MPI.min(MPI.comm_world, abs(x.get_local()).min()) == 1


@skip_in_parallel
def test_near_evaluations(R, mesh):
    # Test that we allow point evaluation that are slightly outside

    u0 = Function(R)
    u0.vector()[:] = 1.0
    a = Vertex(mesh, 0).point().array()
    offset = 0.99 * DOLFIN_EPS

    a_shift_x = Point(a[0] - offset, a[1], a[2]).array()
    assert round(u0(a)[0] - u0(a_shift_x)[0], 7) == 0

    a_shift_xyz = Point(a[0] - offset / math.sqrt(3),
                        a[1] - offset / math.sqrt(3),
                        a[2] - offset / math.sqrt(3)).array()
    assert round(u0(a)[0] - u0(a_shift_xyz)[0], 7) == 0


def test_interpolation_rank1(W):
    @function.expression.numba_eval
    def expr_eval(values, x, cell_idx):
        values[:, 0] = 1.0
        values[:, 1] = 1.0
        values[:, 2] = 1.0

    f = Expression(expr_eval, shape=(3, ))
    w = interpolate(f, W)
    x = w.vector()
    assert abs(x.get_local()).max() == 1
    assert abs(x.get_local()).min() == 1


@pytest.mark.xfail(raises=numba.errors.TypingError)
def test_numba_expression_jit_objmode_fails(W):
    # numba cfunc cannot call into objmode jit function
    @function.expression.numba_eval(numba_jit_options={"forceobj": True})
    def expr_eval(values, x, cell_idx):
        values[0] = 1.0


@pytest.mark.xfail(raises=NotImplementedError)
def test_numba_expression_cfunc_objmode_fails(W):
    # numba does not support cfuncs built in objmode
    @function.expression.numba_eval(numba_cfunc_options={"forceobj": True})
    def expr_eval(values, x, cell_idx):
        values[0] = 1.0


@skip_in_parallel
def test_interpolation_old(V, W, mesh):
    @function.expression.numba_eval
    def expr_eval0(values, x, cell_idx):
        values[:, 0] = 1.0

    @function.expression.numba_eval
    def expr_eval1(values, x, cell_idx):
        values[:, 0] = 1.0
        values[:, 1] = 1.0
        values[:, 2] = 1.0

    # Scalar interpolation
    f0 = Expression(expr_eval0)
    f = Function(V)
    f = interpolate(f0, V)
    assert round(f.vector().norm(cpp.la.Norm.l1) - mesh.num_vertices(), 7) == 0

    # Vector interpolation
    f1 = Expression(expr_eval1, shape=(3, ))
    f = Function(W)
    f.interpolate(f1)
    assert round(f.vector().norm(cpp.la.Norm.l1) - 3 * mesh.num_vertices(),
                 7) == 0


def test_numba_expression_address(V):
    @function.expression.numba_eval
    def expr_eval(values, x, cell_idx):
        values[:, :] = 1.0

    # Handle C func address by hand
    f1 = Expression(expr_eval.address)
    f = Function(V)

    f.interpolate(f1)
    assert (f.vector().get_local() == 1.0).all()
