"""Unit tests for the function library"""

# Copyright (C) 2007-2014 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
from dolfin import *
from math import sin, cos, exp, tan
from numpy import array, zeros, float_
import numpy as np

from dolfin_utils.test import fixture, skip_in_parallel

@fixture
def mesh():
    return UnitCubeMesh(MPI.comm_world, 8, 8, 8)


@fixture
def V(mesh):
    return FunctionSpace(mesh, 'CG', 1)


@fixture
def W(mesh):
    return VectorFunctionSpace(mesh, 'CG', 1)


def test_arbitrary_eval(mesh):
    class F0(UserExpression):
        def eval(self, values, x):
            values[0] = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])

    f0 = F0(name="f0", label="My expression", degree=2)
#    f1 = Expression("a*sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])",
#                    degree=2, a=1., name="f1")
    x = array([0.31, 0.32, 0.33])
    u00 = zeros(1)
    u01 = zeros(1)
    u10 = zeros(1)
    u20 = zeros(1)

    # Check usergeneration of name and label
    assert f0.name() == "f0"
    assert str(f0) == "f0"
    assert f0.label() == "My expression"
    assert f1.name() == "f1"
    assert str(f1) == "f1"
    assert f1.label() == "User defined expression"

    # Check outgeneration of name
    count = int(F0(degree=0).name()[2:])
    assert F0(degree=0).count() == count + 1

    # Test original and vs short evaluation
    f0.eval(u00, x)
    f0(x, values=u01)
    assert round(u00[0] - u01[0], 7) == 0

    # Evaluation with and without return value
    f1(x, values=u10)
    u11 = f1(x)
    assert round(u10[0] - u11, 7) == 0

    # Test *args for coordinate argument
    f1(0.31, 0.32, 0.33, values=u20)
    u21 = f0(0.31, 0.32, 0.33)
    assert round(u20[0] - u21, 7) == 0

    # Test Point evaluation
    p0 = Point(0.31, 0.32, 0.33)
    u21 = f1(p0)
    assert round(u20[0] - u21, 7) == 0

    same_result = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])
    assert round(u00[0] - same_result, 7) == 0
    assert round(u11 - same_result, 7) == 0
    assert round(u21 - same_result, 7) == 0

    # For MUMPS, increase estimated require memory increase. Typically
    # required for small meshes in 3D (solver is called by 'project')
    if has_petsc():
        PETScOptions.set("mat_mumps_icntl_14", 40)

    x = (mesh.coordinates()[0]+mesh.coordinates()[1])/2
#    f2 = Expression("1.0 + 3.0*x[0] + 4.0*x[1] + 0.5*x[2]", degree=2)
    V2 = FunctionSpace(mesh, 'CG', 2)
    g0 = interpolate(f2, V=V2)
    g1 = project(f2, V=V2)

    u3 = f2(x)
    u4 = g0(x)
    u5 = g1(x)
    assert round(u3 - u4, 7) == 0
    assert round(u3 - u5, 4) == 0

    if has_petsc():
        PETScOptions.clear("mat_mumps_icntl_14")


def test_ufl_eval():
    class F0(UserExpression):
        def eval(self, values, x):
            values[0] = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])

    class V0(UserExpression):
        def eval(self, values, x):
            values[0] = x[0]**2
            values[1] = x[1]**2
            values[2] = x[2]**2

        def value_shape(self):
            return (3,)

    f0 = F0(degree=2)
    v0 = V0(degree=2)

    x = (2.0, 1.0, 3.0)

    # Test ufl evaluation through mapping (overriding the Expression
    # with N here):
    def N(x):
        return x[0]**2 + x[1] + 3*x[2]

    assert f0(x, {f0: N}) == 14

    a = f0**2
    b = a(x, {f0: N})
    assert b == 196

    # Test ufl evaluation together with Expression evaluation by dolfin
    # scalar
    assert f0(x) == f0(*x)
    assert (f0**2)(x) == f0(*x)**2
    # vector
    assert all(a == b for a, b in zip(v0(x), v0(*x)))
    assert dot(v0, v0)(x) == sum(v**2 for v in v0(*x))
    assert dot(v0, v0)(x) == 98


def test_overload_and_call_back(V, mesh):
    class F0(UserExpression):
        def eval(self, values, x):
            values[0] = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])

    class F1(UserExpression):
        def __init__(self, mesh, *arg, **kwargs):
            super().__init__(*arg, **kwargs)
            self.mesh = mesh

        def eval_cell(self, values, x, cell):
            c = Cell(self.mesh, cell.index)
            values[0] = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])

    e0 = F0(degree=2)
    e1 = F1(mesh, degree=2)
#    e2 = Expression("sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])", degree=2)

    u0 = interpolate(e0, V)
    u1 = interpolate(e1, V)
    u2 = interpolate(e2, V)

    s0 = norm(u0)
    s1 = norm(u1)
    s2 = norm(u2)

    ref = 0.36557637568519191
    assert round(s0 - ref, 7) == 0
    assert round(s1 - ref, 7) == 0
    assert round(s2 - ref, 7) == 0


def test_wrong_eval():
    # Test wrong evaluation
    class F0(UserExpression):
        def eval(self, values, x):
            values[0] = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])

    f0 = F0(degree=2)
#    f1 = Expression("sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])", degree=2)

    for f in [f0, f1]:
        with pytest.raises(TypeError):
            f("s")
        with pytest.raises(TypeError):
            f([])
        with pytest.raises(TypeError):
            f(0.5, 0.5, 0.5, values=zeros(3, 'i'))
        with pytest.raises(TypeError):
            f([0.3, 0.2, []])
        with pytest.raises(TypeError):
            f(0.3, 0.2, {})
        with pytest.raises(TypeError):
            f(zeros(3), values=zeros(4))
        with pytest.raises(TypeError):
            f(zeros(4), values=zeros(3))


def test_vector_valued_expression_member_function(mesh):
    V = FunctionSpace(mesh,'CG',1)
    W = VectorFunctionSpace(mesh,'CG',1, dim=3)
    fs = [
        Expression(("1", "2", "3"), degree=1),
        Constant((1, 2, 3)),
        interpolate(Constant((1, 2, 3)), W),
    ]
    for f in fs:
        u = Expression("f[0] + f[1] + f[2]", f=f, degree=1)
        v = interpolate(u, V)
        assert np.allclose(v.vector().get_local(), 6.0)
        for g in fs:
            u.f = g
            v = interpolate(u, V)
            assert np.allclose(v.vector().get_local(), 6.0)


def test_no_write_to_const_array():
    class F1(UserExpression):
        def eval(self, values, x):
            x[0] = 1.0
            values[0] = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])

    mesh = UnitCubeMesh(MPI.comm_world, 3, 3, 3)
    f1 = F1(degree=1)
    with pytest.raises(Exception):
        assemble(f1*dx(mesh))


def test_compute_vertex_values(mesh):
    from numpy import zeros, all, array

    e0 = Expression("1", degree=0)
    e1 = Expression(("1", "2", "3"), degree=0)

    e0_values = e0.compute_vertex_values(mesh)
    e1_values = e1.compute_vertex_values(mesh)

    assert all(e0_values == 1)
    assert all(e1_values[:mesh.num_vertices()] == 1)
    assert all(e1_values[mesh.num_vertices():mesh.num_vertices()*2] == 2)
    assert all(e1_values[mesh.num_vertices()*2:mesh.num_vertices()*3] == 3)

@pytest.mark.skip
def test_runtime_exceptions():

    def noDefaultValues():
        Expression("a")

    def wrongDefaultType():
        Expression("a", a="1", degree=1)

    def wrongParameterNames0():
        Expression("foo", bar=1.0, degree=1)

    def wrongParameterNames1():
        Expression("user_parameters", user_parameters=1.0, degree=1)

    with pytest.raises(RuntimeError):
        noDefaultValues()
    with pytest.raises(RuntimeError):
        wrongDefaultType()
    with pytest.raises(RuntimeError):
        wrongParameterNames0()
    with pytest.raises(RuntimeError):
        wrongParameterNames1()


def test_fail_expression_compilation():
    # Compilation failure only happens on one process,
    # and involves a barrier to let the compilation finish
    # before the other processes loads from disk.
    # This tests that a failure can be caught without deadlock.

    def invalidCppExpression():
        Expression("/", degree=0)

    with pytest.raises(RuntimeError):
        invalidCppExpression()


def test_element_instantiation():
    class F0(UserExpression):
        def eval(self, values, x):
            values[0] = 1.0

    class F1(UserExpression):
        def eval(self, values, x):
            values[0] = 1.0
            values[1] = 1.0

        def value_shape(self):
            return (2,)

    class F2(UserExpression):
        def eval(self, values, x):
            values[0] = 1.0
            values[1] = 1.0
            values[2] = 1.0
            values[3] = 1.0

        def value_shape(self):
            return (2, 2)

    e0 = Expression("1", degree=0)
    assert e0.ufl_element().cell() is None

    e1 = Expression("1", cell=triangle, degree=0)
    assert not e1.ufl_element().cell() is None

    e2 = Expression("1", cell=triangle, degree=2)
    assert e2.ufl_element().degree() == 2

    e3 = Expression(["1", "1"], cell=triangle, degree=0)
    assert isinstance(e3.ufl_element(), VectorElement)

    e4 = Expression((("1", "1"), ("1", "1")), cell=triangle, degree=0)
    assert isinstance(e4.ufl_element(), TensorElement)

    f0 = F0(degree=0)
    assert f0.ufl_element().cell() is None

    f1 = F0(cell=triangle, degree=0)
    assert not f1.ufl_element().cell() is None

    f2 = F0(cell=triangle, degree=2)
    assert f2.ufl_element().degree() == 2

    f3 = F1(cell=triangle, degree=0)
    assert isinstance(f3.ufl_element(), VectorElement)

    f4 = F2(cell=triangle, degree=0)
    assert isinstance(f4.ufl_element(), TensorElement)


def test_num_literal():
    e0 = Expression("1e10", degree=0)
    assert e0(0, 0, 0) == 1e10

    e1 = Expression("1e-10", degree=0)
    assert e1(0, 0, 0) == 1e-10

    e2 = Expression("1e+10", degree=0)
    assert e2(0, 0, 0) == 1e+10

    e3 = Expression(".5", degree=0)
    assert e3(0, 0, 0) == 0.5

    e4 = Expression("x[0] * sin(.5)", degree=2)
    assert e4(0, 0, 0) == 0.

    e5 = Expression(["2*t0", "-t0"], t0=1.0, degree=0)
    values = e5(0, 0, 0)
    assert values[0] == 2.
    assert values[1] == -1.


def test_name_space_usage(mesh):
    e0 = Expression("std::sin(x[0])*cos(x[1])", degree=2)
    e1 = Expression("sin(x[0])*std::cos(x[1])", degree=2)
    assert round(assemble(e0*dx(mesh)) - assemble(e1*dx(mesh)), 7) == 0


def test_generic_function_attributes(mesh, V):
    tc = Constant(2.0)
    te = Expression("value", value=tc, degree=0)

    assert round(tc(0) - te(0), 7) == 0
    tc.assign(1.0)
    assert round(tc(0) - te(0), 7) == 0

    tf = Function(V)
    tf.vector()[:] = 1.0

    e0 = Expression(["2*t", "-t"], t=tc, degree=0)
    e1 = Expression(["2*t0", "-t0"], t0=1.0, degree=0)
    e2 = Expression("t", t=te, degree=0)
    e3 = Expression("t", t=tf, degree=0)

    assert (round(assemble(inner(e0, e0)*dx(mesh)) -
                  assemble(inner(e1, e1)*dx(mesh)), 7) == 0)

    assert (round(assemble(inner(e2, e2)*dx(mesh)) -
                  assemble(inner(e3, e3)*dx(mesh)), 7) == 0)

    tc.assign(3.0)
    e1.t0 = float(tc)

    assert (round(assemble(inner(e0, e0)*dx(mesh)) -
                  assemble(inner(e1, e1)*dx(mesh)), 7) == 0)

    tc.assign(5.0)

    assert assemble(inner(e2, e2)*dx(mesh)) != assemble(inner(e3, e3)*dx(mesh))

    assert (round(assemble(e0[0]*dx(mesh)) -
                  assemble(2*e2*dx(mesh)), 7) == 0)

    e2.t = e3.t

    assert (round(assemble(inner(e2, e2)*dx(mesh)) -
                  assemble(inner(e3, e3)*dx(mesh)), 7) == 0)

    W = FunctionSpace(mesh, V.ufl_element()*V.ufl_element())

    # Test wrong kwargs
    with pytest.raises(Exception):
        Expression("t", t=mesh, degree=0)
    with pytest.raises(Exception):
        Expression("t", t=W, degree=0)

    # Test non-scalar GenericFunction
    f2 = Function(W)
    e2.t = f2

    with pytest.raises(RuntimeError):
        e2(0, 0)

    # Test user_parameters assignment
    assert "value" in te.user_parameters
    te.user_parameters["value"] = Constant(5.0)
    assert te(0.0) == 5.0

    te.user_parameters.update(dict(value=Constant(3.0)))
    assert te(0.0) == 3.0

    te.user_parameters.update([("value", Constant(4.0))])
    assert te(0.0) == 4.0

    # Test wrong assignment
    with pytest.raises(Exception):
        te.user_parameters.__setitem__("value", 1.0)
    with pytest.raises(KeyError):
        te.user_parameters.__setitem__("values", 1.0)


def test_doc_string_eval():
    """
    This test tests all features documented in the doc string of
    Expression. If this test breaks and it is fixed the corresponding fixes
    need also be updated in the docstring.
    """

    square = UnitSquareMesh(10, 10)
    V = VectorFunctionSpace(square, "CG", 1)

    f0 = Expression('sin(x[0]) + cos(x[1])', degree=1)
    f1 = Expression(('cos(x[0])', 'sin(x[1])'), element=V.ufl_element())
    assert round(f0(0, 0) - sum(f1(0, 0)), 7) == 0

    f2 = Expression((('exp(x[0])', 'sin(x[1])'),
                     ('sin(x[0])', 'tan(x[1])')), degree=1)
    assert round(sum(f2(0, 0)) - 1.0, 7) == 0

    f = Expression('A*sin(x[0]) + B*cos(x[1])', A=2.0, B=Constant(4.0),
                   degree=2)
    assert round(f(pi/4, pi/4) - 6./sqrt(2), 7) == 0

    f.A = 5.0
    f.B = Expression("value", value=6.0, degree=0)
    assert round(f(pi/4, pi/4) - 11./sqrt(2), 7) == 0

    f.user_parameters["A"] = 1.0
    f.user_parameters["B"] = Constant(5.0)
    assert round(f(pi/4, pi/4) - 6./sqrt(2), 7) == 0



def test_doc_string_python_expressions(mesh):
    """This test tests all features documented in the doc string of
    Expression. If this test breaks and it is fixed the corresponding
    fixes need also be updated in the docstring.

    """

    square = UnitSquareMesh(4, 4)

    class MyExpression0(UserExpression):
        def eval(self, value, x):
            dx = x[0] - 0.5
            dy = x[1] - 0.5
            value[0] = 500.0*exp(-(dx*dx + dy*dy)/0.02)
            value[1] = 250.0*exp(-(dx*dx + dy*dy)/0.01)

        def value_shape(self):
            return (2,)

    f0 = MyExpression0(degree=2)
    values = f0(0.2, 0.3)
    dx = 0.2 - 0.5
    dy = 0.3 - 0.5

    assert round(values[0] - 500.0*exp(-(dx*dx + dy*dy)/0.02), 7) == 0
    assert round(values[1] - 250.0*exp(-(dx*dx + dy*dy)/0.01), 7) == 0

    ufc_cell_attrs = ["cell_shape", "index", "topological_dimension",
                      "geometric_dimension", "local_facet", "mesh_identifier"]

    class MyExpression1(UserExpression):
        def eval_cell(self_expr, value, x, ufc_cell):
            # Check attributes in ufc cell
            for attr in ufc_cell_attrs:
                  assert hasattr(ufc_cell, attr)

            if ufc_cell.index > 10:
                value[0] = 1.0
            else:
                value[0] = -1.0

    f1 = MyExpression1(degree=0)
    assemble(f1*ds(square))

    class MyExpression2(UserExpression):
        def __init__(self, mesh, domain, *arg, **kwargs):
            super().__init__(*arg, **kwargs)
            self._mesh = mesh
            self._domain = domain

        def eval(self, values, x):
            pass

    cell_data = MeshFunction('size_t', square, square.topology().dim())

    P1 = FiniteElement("Lagrange", square.ufl_cell(), 1)
    f3 = MyExpression2(square, cell_data, element=P1)

    assert id(f3._mesh) == id(square)
    assert id(f3._domain) == id(cell_data)
