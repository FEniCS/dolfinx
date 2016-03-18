#!/usr/bin/env py.test

"""Unit tests for the Function class"""

# Copyright (C) 2011-2014 Garth N. Wells
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

import pytest
from dolfin import *
import ufl

from dolfin_utils.test import *

@fixture
def mesh():
    return UnitCubeMesh(8, 8, 8)

@fixture
def R(mesh):
    return FunctionSpace(mesh, 'R', 0)

@fixture
def V(mesh):
    return FunctionSpace(mesh, 'CG', 1)

@fixture
def W(mesh):
    return VectorFunctionSpace(mesh, 'CG', 1)


def test_name_argument(W):
    u = Function(W)
    v = Function(W, name="v")
    assert u.name() == "f_%d" % u.count()
    assert v.name() == "v"
    assert str(v) == "v"

def test_in_function_space(W):
    u = Function(W)
    v = Function(W)
    assert u in W
    assert u in u.function_space()
    assert u in v.function_space()
    for i, usub in enumerate(u.split()):
        assert usub in W.sub(i)

def test_compute_vertex_values(V, W, mesh):
    from numpy import zeros, all, array
    u = Function(V)
    v = Function(W)

    u.vector()[:] = 1.
    v.vector()[:] = 1.

    u_values = u.compute_vertex_values(mesh)
    v_values = v.compute_vertex_values(mesh)

    assert all(u_values==1)

def test_assign(V, W):
    from ufl.algorithms import replace

    for V0, V1, vector_space in [(V, W, False), (W, V, True)]:
        u = Function(V0)
        u0 = Function(V0)
        u1 = Function(V0)
        u2 = Function(V0)
        u3 = Function(V1)

        u.vector()[:] =  1.0
        u0.vector()[:] = 2.0
        u1.vector()[:] = 3.0
        u2.vector()[:] = 4.0
        u3.vector()[:] = 5.0

        scalars = {u:1.0, u0:2.0, u1:3.0, u2:4.0, u3:5.0}

        uu = Function(V0)
        uu.assign(2*u)
        assert uu.vector().sum() == u0.vector().sum()

        uu = Function(V1)
        uu.assign(3*u)
        assert uu.vector().sum() == u1.vector().sum()

        # Test complex assignment
        expr = 3*u-4*u1-0.1*4*u*4+u2+3*u0/3./0.5
        expr_scalar = 3-4*3-0.1*4*4+4.+3*2./3./0.5
        uu.assign(expr)
        assert round(uu.vector().sum() -\
                      float(expr_scalar*uu.vector().size()), 7) == 0

        # Test expression scaling
        expr = 3*expr
        expr_scalar *= 3
        uu.assign(expr)
        assert round(uu.vector().sum() - \
                      float(expr_scalar*uu.vector().size()), 7) == 0

        # Test expression scaling
        expr = expr/4.5
        expr_scalar /= 4.5
        uu.assign(expr)
        assert round(uu.vector().sum() - \
                      float(expr_scalar*uu.vector().size()), 7) == 0

        # Test self assignment
        expr = 3*u - Constant(5)*u2 + u1 - 5*u
        expr_scalar = 3 - 5*4. + 3. - 5
        u.assign(expr)
        assert round(u.vector().sum() - \
                      float(expr_scalar*u.vector().size()), 7) == 0

        # Test zero assignment
        u.assign(-u2/2+2*u1-u1/0.5+u2*0.5)
        assert round(u.vector().sum() - 0.0, 7) == 0

        # Test errounious assignments
        uu = Function(V1)
        f = Expression("1.0", degree=0)
        with pytest.raises(RuntimeError):
            uu.assign(1.0)
        with pytest.raises(RuntimeError):
            uu.assign(4*f)

        if not vector_space:
            with pytest.raises(RuntimeError):
                uu.assign(u*u0)
            with pytest.raises(RuntimeError):
                uu.assign(4/u0)
            with pytest.raises(RuntimeError):
                uu.assign(4*u*u1)


def test_axpy(V, W):
    for V0, V1, vector_space in [(V, W, False), (W, V, True)]:
        u = Function(V0)
        u0 = Function(V0)
        u1 = Function(V0)
        u2 = Function(V0)
        u3 = Function(V1)

        u.vector()[:] =  1.0
        u0.vector()[:] = 2.0
        u1.vector()[:] = 3.0
        u2.vector()[:] = 4.0
        u3.vector()[:] = 5.0

        axpy = FunctionAXPY(u1, 2.0)
        u.assign(axpy)
        expr_scalar = 3*2

        assert round(u.vector().sum() - \
                      float(expr_scalar*u.vector().size()), 7) == 0

        axpy = FunctionAXPY([(2.0, u1), (3.0, u2)])

        u.assign(axpy)
        expr_scalar = 3*2+3*4.0

        assert round(u.vector().sum() - \
                     float(expr_scalar*u.vector().size()), 7) == 0

        axpy = axpy*3
        u.assign(axpy)
        expr_scalar *= 3

        assert round(u.vector().sum() - \
                      float(expr_scalar*u.vector().size()), 7) == 0

        axpy0 = axpy/5
        u.assign(axpy0)
        expr_scalar0 = expr_scalar/5

        assert round(u.vector().sum() - \
                      float(expr_scalar0*u.vector().size()), 7) == 0

        axpy1 = axpy0+axpy
        u.assign(axpy1)
        expr_scalar1 = expr_scalar0 + expr_scalar

        assert round(u.vector().sum() - \
                      float(expr_scalar1*u.vector().size()), 7) == 0

        axpy1 = axpy0-axpy
        u.assign(axpy1)
        expr_scalar1 = expr_scalar0 - expr_scalar

        assert round(u.vector().sum() - \
                      float(expr_scalar1*u.vector().size()), 7) == 0

        axpy1 = axpy0+u1
        u.assign(axpy1)
        expr_scalar1 = expr_scalar0 + 3.0

        assert round(u.vector().sum() - \
                      float(expr_scalar1*u.vector().size()), 7) == 0

        axpy1 = axpy0-u2
        u.assign(axpy1)
        expr_scalar1 = expr_scalar0 - 4.0

        assert round(u.vector().sum() - \
                      float(expr_scalar1*u.vector().size()), 7) == 0

        with pytest.raises(RuntimeError):
            FunctionAXPY(u, u3, 0)

        axpy = FunctionAXPY(u3, 2.0)

        with pytest.raises(RuntimeError):
            axpy+u

def test_call(R, V, W, mesh):
    from numpy import zeros, all, array
    u0 = Function(R)
    u1 = Function(V)
    u2 = Function(W)
    e0=Expression("x[0] + x[1] + x[2]", degree=1)
    e1=Expression(("x[0] + x[1] + x[2]", "x[0] - x[1] - x[2]", "x[0] + x[1] + x[2]"), degree=1)

    u0.vector()[:] = 1.0
    u1.interpolate(e0)
    u2.interpolate(e1)

    p0 = (Vertex(mesh,0).point()+Vertex(mesh,1).point())/2
    x0 = (mesh.coordinates()[0]+mesh.coordinates()[1])/2
    x1 = tuple(x0)

    assert round(u0(*x1) - u0(x0), 7) == 0
    assert round(u0(x1) - u0(p0), 7) == 0
    assert round(u1(x1) - u1(x0), 7) == 0
    assert round(u1(*x1) - u1(p0), 7) == 0
    assert round(u2(x1)[0] - u1(p0), 7) == 0

    assert all(u2(*x1) == u2(x0))
    assert all(u2(*x1) == u2(p0))

    values = zeros(mesh.geometry().dim(), dtype='d')
    u2(p0, values=values)
    assert all(values == u2(x0))

    with pytest.raises(TypeError):
        u0([0,0,0,0])
    with pytest.raises(TypeError):
        u0([0,0])

def test_constant_float_conversion():
    c = Constant(3.45)
    assert float(c) == 3.45

def test_real_function_float_conversion1(R):
    c = Function(R)
    assert float(c) == 0.0

def test_real_function_float_conversion2(R):
    c = Function(R)
    c.assign(Constant(2.34))
    assert float(c) == 2.34

def test_real_function_float_conversion3(R):
    c = Function(R)
    c.vector()[:] = 1.23
    assert float(c) == 1.23

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
    # Note that 1.5 < 0 == False == 1.5 < 1, but that is not what we compare here:
    assert not (c < 0) == (c < 1)
    # This protects from "if c < 0: ..." misuse:
    with pytest.raises(ufl.UFLException):
        bool(c < 0)
    with pytest.raises(ufl.UFLException):
        not c < 0

def test_interpolation_mismatch_rank0(W):
    f = Expression("1.0", degree=0)
    with pytest.raises(RuntimeError):
        interpolate(f, W)

def test_interpolation_mismatch_rank1(W):
    f = Expression(("1.0", "1.0"), degree=0)
    with pytest.raises(RuntimeError):
        interpolate(f, W)

def test_interpolation_jit_rank0(V):
    f = Expression("1.0", degree=0)
    w = interpolate(f, V)
    x = w.vector()
    assert x.max() == 1
    assert x.min() == 1

@skip_in_parallel
def test_extrapolation(V):
    original_parameters = parameters["allow_extrapolation"]

    f0 = Function(V)
    with pytest.raises(RuntimeError):
        f0.__call__((0., 0, -1))

    mesh1 = UnitSquareMesh(3, 3)
    V1 = FunctionSpace(mesh1, "CG", 1)

    mesh2 = UnitTriangleMesh()
    V2 = FunctionSpace(mesh2, "CG", 1)

    parameters["allow_extrapolation"] = True
    f1 = Function(V1)
    f1.vector()[:] = 1.0
    assert round(f1(0.,-1) - 1.0, 7) == 0

    parameters["allow_extrapolation"] = False
    f2 = Function(V2)
    with pytest.raises(RuntimeError):
        f2.__call__((0.,-1.))

    parameters["allow_extrapolation"] = True
    f3 = Function(V2)
    f3.vector()[:] = 1.0
    assert round(f3(0.,-1) - 1.0, 7) == 0

    parameters["allow_extrapolation"] = original_parameters

    f1 = Function(V1)
    f1.set_allow_extrapolation(True)
    f1.vector()[:] = 1.0
    assert round(f1(0.,-1) - 1.0, 7) == 0

    f2 = Function(V2)

    f2.set_allow_extrapolation(False)
    with pytest.raises(RuntimeError):
        f2.__call__((0.,-1.))

    f2.set_allow_extrapolation(True)
    f2.vector()[:] = 1.0
    assert round(f2(0.,-1) - 1.0, 7) == 0

    f2.set_allow_extrapolation(True)
    assert f2.get_allow_extrapolation() is True
    f2.set_allow_extrapolation(False)
    assert f2.get_allow_extrapolation() is False

    parameters["allow_extrapolation"] = original_parameters


def test_interpolation_jit_rank1(W):
    f = Expression(("1.0", "1.0", "1.0"), degree=0)
    w = interpolate(f, W)
    x = w.vector()
    assert x.max() == 1
    assert x.min() == 1


@skip_in_parallel
def test_interpolation_old(V, W, mesh):

    class F0(Expression):
        def eval(self, values, x):
            values[0] = 1.0

    class F1(Expression):
        def eval(self, values, x):
            values[0] = 1.0
            values[1] = 1.0
            values[2] = 1.0
        def value_shape(self):
            return (3,)

    # Scalar interpolation
    f0 = F0(degree=0)
    f = Function(V)
    f.interpolate(f0)
    assert round(f.vector().norm("l1") - mesh.num_vertices(), 7) == 0

    # Vector interpolation
    f1 = F1(degree=0)
    f = Function(W)
    f.interpolate(f1)
    assert round(f.vector().norm("l1") - 3*mesh.num_vertices(), 7) == 0
