# Copyright (C) 2011-2014 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the Function class"""

import importlib
import math

import cffi
import numpy as np
import pytest
from petsc4py import PETSc

import ufl
from dolfin import (MPI, Function, FunctionSpace, TensorFunctionSpace,
                    UnitCubeMesh, VectorFunctionSpace, cpp, geometry)
from dolfin_utils.test.skips import skip_if_complex, skip_in_parallel


@pytest.fixture
def mesh():
    return UnitCubeMesh(MPI.comm_world, 3, 3, 3)


@pytest.fixture
def R(mesh):
    return FunctionSpace(mesh, ('R', 0))


@pytest.fixture
def V(mesh):
    return FunctionSpace(mesh, ('CG', 1))


@pytest.fixture
def W(mesh):
    return VectorFunctionSpace(mesh, ('CG', 1))


@pytest.fixture
def Q(mesh):
    return TensorFunctionSpace(mesh, ('CG', 1))


def test_name_argument(W):
    u = Function(W)
    v = Function(W, name="v")
    assert u.name == "f_{}".format(u.count())
    assert v.name == "v"
    assert str(v) == "v"


def test_compute_point_values(V, W, mesh):
    u = Function(V)
    v = Function(W)
    with u.vector.localForm() as u_local, v.vector.localForm() as v_local:
        u_local.set(1.0)
        v_local.set(1.0)
    u_values = u.compute_point_values()
    v_values = v.compute_point_values()

    u_ones = np.ones_like(u_values, dtype=np.float64)
    assert np.all(np.isclose(u_values, u_ones))
    v_ones = np.ones_like(v_values, dtype=np.float64)
    assert np.all(np.isclose(v_values, v_ones))
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

        u.vector[:] = 1.0
        u0.vector[:] = 2.0
        u1.vector[:] = 3.0
        u2.vector[:] = 4.0
        u3.vector[:] = 5.0

        uu = Function(V0)
        uu.assign(2 * u)
        assert uu.vector.get_local().sum() == u0.vector.get_local().sum()

        uu = Function(V1)
        uu.assign(3 * u)
        assert uu.vector.get_local().sum() == u1.vector.get_local().sum()

        # Test complex assignment
        expr = 3 * u - 4 * u1 - 0.1 * 4 * u * 4 + u2 + 3 * u0 / 3. / 0.5
        expr_scalar = 3 - 4 * 3 - 0.1 * 4 * 4 + 4. + 3 * 2. / 3. / 0.5
        uu.assign(expr)
        assert (round(
            uu.vector.get_local().sum() - float(
                expr_scalar * uu.vector.size()), 7) == 0)

        # Test self assignment
        expr = 3 * u - 5.0 * u2 + u1 - 5 * u
        expr_scalar = 3 - 5 * 4. + 3. - 5
        u.assign(expr)
        assert (round(
            u.vector.get_local().sum() - float(
                expr_scalar * u.vector.size()), 7) == 0)

        # Test zero assignment
        u.assign(-u2 / 2 + 2 * u1 - u1 / 0.5 + u2 * 0.5)
        assert round(u.vector.get_local().sum() - 0.0, 7) == 0

        # Test erroneous assignments
        uu = Function(V1)

        def f(values, x):
            values[:, 0] = 1.0

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


def test_eval(R, V, W, Q, mesh):
    u0 = Function(R)
    u1 = Function(V)
    u2 = Function(W)
    u3 = Function(Q)

    def e1(x):
        return x[:, 0] + x[:, 1] + x[:, 2]

    def e2(x):
        values = np.empty((x.shape[0], 3))
        values[:, 0] = x[:, 0] + x[:, 1] + x[:, 2]
        values[:, 1] = x[:, 0] - x[:, 1] - x[:, 2]
        values[:, 2] = x[:, 0] + x[:, 1] + x[:, 2]
        return values

    def e3(x):
        values = np.empty((x.shape[0], 9))
        values[:, 0] = x[:, 0] + x[:, 1] + x[:, 2]
        values[:, 1] = x[:, 0] - x[:, 1] - x[:, 2]
        values[:, 2] = x[:, 0] + x[:, 1] + x[:, 2]
        values[:, 3] = x[:, 0]
        values[:, 4] = x[:, 1]
        values[:, 5] = x[:, 2]
        values[:, 6] = -x[:, 0]
        values[:, 7] = -x[:, 1]
        values[:, 8] = -x[:, 2]
        return values

    u0.vector.set(1.0)
    u1.interpolate(e1)
    u2.interpolate(e2)
    u3.interpolate(e3)

    x0 = (mesh.geometry.x(0) + mesh.geometry.x(1)) / 2.0
    tree = geometry.BoundingBoxTree(mesh, mesh.geometry.dim)
    cells = geometry.compute_first_entity_collision(tree, mesh, x0)
    assert np.allclose(u3.eval(x0, cells)[:3], u2.eval(x0, cells), rtol=1e-15, atol=1e-15)
    with pytest.raises(ValueError):
        u0.eval([0, 0, 0, 0], 0)
    with pytest.raises(ValueError):
        u0.eval([0, 0], 0)


<<<<<<< HEAD
=======
def test_eval_reference(V, mesh):
    u = Function(V)

    u.interpolate(lambda x: x[:, 0])

    # Get reference coordinates of dofs and evaluate u
    X = V.element.dof_reference_coordinates()
    values0 = u.eval_reference(X)

    # FIXME: This step shpuld be a lot simpler
    # Compute physical coordinate of X
    coord_dofs = mesh.coordinate_dofs().entity_points()
    x_g = mesh.geometry.points
    cmap = V.mesh.geometry.coord_mapping
    x_dofs = []
    for c in range(mesh.num_cells()):
        x_coord_new = np.zeros([4, 3])
        for v in range(4):
            x_coord_new[v] = x_g[coord_dofs[c, v]]
        x = X.copy()
        cmap.push_forward(x, X, x_coord_new)
        for _x in x:
            x_dofs.append(_x)

    x = np.asarray(x_dofs)
    cells = [c for c in range(mesh.num_cells()) for j in range(4)]
    values1 = u.eval(x, np.asarray(cells))

    assert values1 == pytest.approx(values0)


>>>>>>> acf07c641... Use standard push-forward jargon.
def test_eval_multiple(W):
    u = Function(W)
    u.vector.set(1.0)
    mesh = W.mesh
    x0 = (mesh.geometry.x(0) + mesh.geometry.x(1)) / 2.0
    x = np.array([x0, x0 + 1.0e8])
    tree = geometry.BoundingBoxTree(mesh, W.mesh.geometry.dim)
    cells = geometry.compute_first_entity_collision(tree, mesh, x)
    u.eval(x[0], cells[0])


def test_scalar_conditions(R):
    c = Function(R)
    c.vector.set(1.5)

    # Float conversion does not interfere with boolean ufl expressions
    assert isinstance(ufl.lt(c, 3), ufl.classes.LT)
    assert not isinstance(ufl.lt(c, 3), bool)

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


@pytest.mark.skip
def test_interpolation_mismatch_rank0(W):
    def f(x):
        return np.ones(x.shape[0])
    u = Function(W)
    with pytest.raises(RuntimeError):
        u.interpolate(f)


@pytest.mark.skip
def test_interpolation_mismatch_rank1(W):
    def f(values, x):
        return np.ones((x.shape[0], 2))

    u = Function(W)
    with pytest.raises(RuntimeError):
        u.interpolate(f)


def test_interpolation_rank0(V):
    class MyExpression:
        def __init__(self):
            self.t = 0.0

        def eval(self, x):
            return np.full(x.shape[0], self.t)

    f = MyExpression()
    f.t = 1.0
    w = Function(V)
    w.interpolate(f.eval)
    with w.vector.localForm() as x:
        assert (x[:] == 1.0).all()
    f.t = 2.0
    w.interpolate(f.eval)
    with w.vector.localForm() as x:
        assert (x[:] == 2.0).all()


@skip_in_parallel
def xtest_near_evaluations(R, mesh):
    # Test that we allow point evaluation that are slightly outside
    bb_tree = cpp.geometry.BoundingBoxTree(mesh, mesh.geometry.dim)
    u0 = Function(R)
    u0.vector.set(1.0)
    a = mesh.geometry.x(0)
    offset = 0.99 * np.finfo(float).eps

    a_shift_x = np.array([a[0] - offset, a[1], a[2]])
    assert u0.eval(a, bb_tree)[0] == pytest.approx(u0.eval(a_shift_x, bb_tree)[0])

    a_shift_xyz = np.array([a[0] - offset / math.sqrt(3),
                            a[1] - offset / math.sqrt(3),
                            a[2] - offset / math.sqrt(3)])
    assert u0.eval(a, bb_tree)[0] == pytest.approx(u0.eval(a_shift_xyz, bb_tree)[0])


def test_interpolation_rank1(W):
    def f(x):
        values = np.empty((x.shape[0], 3))
        values[:, 0] = 1.0
        values[:, 1] = 1.0
        values[:, 2] = 1.0
        return values

    w = Function(W)
    w.interpolate(f)
    x = w.vector
    assert x.max()[1] == 1.0
    assert x.min()[1] == 1.0


@skip_in_parallel
def test_interpolation_old(V, W, mesh):
    def f0(x):
        return np.ones(x.shape[0])

    def f1(x):
        return np.ones((x.shape[0], mesh.geometry.dim))

    # Scalar interpolation
    f = Function(V)
    f.interpolate(f0)
    assert round(f.vector.norm(PETSc.NormType.N1) - mesh.num_entities(0),
                 7) == 0

    # Vector interpolation
    f = Function(W)
    f.interpolate(f1)
    assert round(f.vector.norm(PETSc.NormType.N1) - 3 * mesh.num_entities(0),
                 7) == 0


@skip_if_complex
def test_cffi_expression(V):
    code_h = """
    void eval(double* values, int num_points, int value_size, const double* x, int gdim);
    """

    code_c = """
    void eval(double* values, int num_points, int value_size, const double* x, int gdim)
    {
      for (int i = 0; i < num_points; ++i)
        values[i*value_size + 0] = x[i*gdim + 0] + x[i*gdim + 1];
    }
    """
    module = "_expr_eval" + str(MPI.comm_world.rank)

    # Build the kernel
    ffi = cffi.FFI()
    ffi.set_source(module, code_c)
    ffi.cdef(code_h)
    ffi.compile()

    # Import the compiled kernel
    kernel_mod = importlib.import_module(module)
    ffi, lib = kernel_mod.ffi, kernel_mod.lib

    # Get pointer to the compiled function
    eval_ptr = ffi.cast("uintptr_t", ffi.addressof(lib, "eval"))

    # Handle C func address by hand
    f1 = Function(V)
    f1.interpolate(int(eval_ptr))

    def expr_eval2(x):
        return x[:, 0] + x[:, 1]

    f2 = Function(V)
    f2.interpolate(expr_eval2)
    assert (f1.vector - f2.vector).norm() < 1.0e-12
