# Copyright (C) 2011-2021 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the Function class"""

import importlib

import cffi
import dolfinx
import numpy as np
import pytest
import ufl
from dolfinx import (Function, FunctionSpace, TensorFunctionSpace,
                     UnitCubeMesh, VectorFunctionSpace, geometry)
from dolfinx.mesh import create_mesh
from dolfinx_utils.test.skips import skip_if_complex, skip_in_parallel
from mpi4py import MPI
from petsc4py import PETSc


@pytest.fixture
def mesh():
    return UnitCubeMesh(MPI.COMM_WORLD, 3, 3, 3)


@pytest.fixture
def V(mesh):
    return FunctionSpace(mesh, ('Lagrange', 1))


@pytest.fixture
def W(mesh):
    return VectorFunctionSpace(mesh, ('Lagrange', 1))


@pytest.fixture
def Q(mesh):
    return TensorFunctionSpace(mesh, ('Lagrange', 1))


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


def test_assign(V, W):
    for V_ in [V, W]:
        u = Function(V_)
        u0 = Function(V_)
        u1 = Function(V_)
        u2 = Function(V_)
        with u.vector.localForm() as loc:
            loc.set(1)
        with u0.vector.localForm() as loc:
            loc.set(2)
        with u1.vector.localForm() as loc:
            loc.set(3)
        with u2.vector.localForm() as loc:
            loc.set(4)

        # Test assign + scale
        uu = Function(V_)
        u.vector.copy(result=uu.vector)
        uu.vector.scale(2)
        assert uu.vector.array.sum() == u0.vector.array.sum()

        # Test complex assignment
        expr = 3 * u.vector - 4 * u1.vector - 0.1 * 4 * u.vector * 4 + u2.vector + 3 * u0.vector / 3. / 0.5
        expr.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        expr_scalar = 3 - 4 * 3 - 0.1 * 4 * 4 + 4. + 3 * 2. / 3. / 0.5

        expr.copy(result=uu.vector)
        uu.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        assert np.isclose(uu.vector.array.sum() - expr_scalar * uu.vector.local_size, 0)

        # Test self assignment
        expr = 3 * u.vector - 5.0 * u2.vector + u1.vector - 5 * u.vector
        expr.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        expr_scalar = 3 - 5 * 4. + 3. - 5
        expr.copy(result=u.vector)
        u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        assert np.isclose(u.vector.array.sum() - expr_scalar * u.vector.local_size, 0)

        # Test zero assignment
        expr = -u2.vector / 2 + 2 * u1.vector - u1.vector / 0.5 + u2.vector * 0.5
        expr.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        expr.copy(result=u.vector)
        u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        assert round(u.vector.array.sum() - 0.0, 7) == 0


def test_eval(V, W, Q, mesh):
    u1 = Function(V)
    u2 = Function(W)
    u3 = Function(Q)

    def e1(x):
        return x[0] + x[1] + x[2]

    def e2(x):
        values = np.empty((3, x.shape[1]))
        values[0] = x[0] + x[1] + x[2]
        values[1] = x[0] - x[1] - x[2]
        values[2] = x[0] + x[1] + x[2]
        return values

    def e3(x):
        values = np.empty((9, x.shape[1]))
        values[0] = x[0] + x[1] + x[2]
        values[1] = x[0] - x[1] - x[2]
        values[2] = x[0] + x[1] + x[2]
        values[3] = x[0]
        values[4] = x[1]
        values[5] = x[2]
        values[6] = -x[0]
        values[7] = -x[1]
        values[8] = -x[2]
        return values

    u1.interpolate(e1)
    u2.interpolate(e2)
    u3.interpolate(e3)

    x0 = (mesh.geometry.x[0] + mesh.geometry.x[1]) / 2.0
    tree = geometry.BoundingBoxTree(mesh, mesh.geometry.dim)
    cell_candidates = geometry.compute_collisions_point(tree, x0)
    cell = dolfinx.cpp.geometry.select_colliding_cells(mesh, cell_candidates, x0, 1)

    assert np.allclose(u3.eval(x0, cell)[:3], u2.eval(x0, cell), rtol=1e-15, atol=1e-15)


def test_eval_multiple(W):
    u = Function(W)
    u.vector.set(1.0)
    mesh = W.mesh
    x0 = (mesh.geometry.x[0] + mesh.geometry.x[1]) / 2.0
    x = np.array([x0, x0 + 1.0e8])
    tree = geometry.BoundingBoxTree(mesh, mesh.geometry.dim)
    cell_candidates = [geometry.compute_collisions_point(tree, xi) for xi in x]
    assert len(cell_candidates[1]) == 0
    cell_candidates = cell_candidates[0]
    cell = dolfinx.cpp.geometry.select_colliding_cells(mesh, cell_candidates, x0, 1)

    u.eval(x[0], cell)


@skip_in_parallel
def test_eval_manifold():
    # Simple two-triangle surface in 3d
    vertices = [(0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0,
                                                                    0.0)]
    cells = [(0, 1, 2), (0, 1, 3)]
    cell = ufl.Cell("triangle", geometric_dimension=3)
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 1))
    mesh = create_mesh(MPI.COMM_WORLD, cells, vertices, domain)
    Q = FunctionSpace(mesh, ("Lagrange", 1))
    u = Function(Q)
    u.interpolate(lambda x: x[0] + x[1])
    assert np.isclose(u.eval([0.75, 0.25, 0.5], 0)[0], 1.0)


def test_interpolation_mismatch_rank0(W):
    def f(x):
        return np.ones(x.shape[1])
    u = Function(W)
    with pytest.raises(RuntimeError):
        u.interpolate(f)


def test_interpolation_mismatch_rank1(W):
    def f(x):
        return np.ones((2, x.shape[1]))

    u = Function(W)
    with pytest.raises(RuntimeError):
        u.interpolate(f)


def test_mixed_element_interpolation():
    def f(x):
        return np.ones(2, x.shape[1])
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 3, 3, 3)
    el = ufl.FiniteElement("CG", mesh.ufl_cell(), 1)
    V = dolfinx.FunctionSpace(mesh, ufl.MixedElement([el, el]))
    u = dolfinx.Function(V)
    with pytest.raises(RuntimeError):
        u.interpolate(f)


def test_interpolation_rank0(V):
    class MyExpression:
        def __init__(self):
            self.t = 0.0

        def eval(self, x):
            return np.full(x.shape[1], self.t)

    f = MyExpression()
    f.t = 1.0
    w = Function(V)
    w.interpolate(f.eval)
    with w.vector.localForm() as x:
        assert (x[:] == 1.0).all()

    num_vertices = V.mesh.topology.index_map(0).size_global
    assert np.isclose(w.vector.norm(PETSc.NormType.N1) - num_vertices, 0)

    f.t = 2.0
    w.interpolate(f.eval)
    with w.vector.localForm() as x:
        assert (x[:] == 2.0).all()


def test_interpolation_rank1(W):
    def f(x):
        values = np.empty((3, x.shape[1]))
        values[0] = 1.0
        values[1] = 1.0
        values[2] = 1.0
        return values

    w = Function(W)
    w.interpolate(f)
    x = w.vector
    assert x.max()[1] == 1.0
    assert x.min()[1] == 1.0

    num_vertices = W.mesh.topology.index_map(0).size_global
    assert round(w.vector.norm(PETSc.NormType.N1) - 3 * num_vertices, 7) == 0


@skip_if_complex
def test_cffi_expression(V):
    code_h = """
    void eval(double* values, int num_points, int value_size, const double* x);
    """

    code_c = """
    void eval(double* values, int num_points, int value_size, const double* x)
    {
      for (int i = 0; i < num_points; ++i)
        values[i*value_size + 0] = x[i*3 + 0] + x[i*3 + 1];
    }
    """
    module = "_expr_eval" + str(MPI.COMM_WORLD.rank)

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
        return x[0] + x[1]

    f2 = Function(V)
    f2.interpolate(expr_eval2)
    assert (f1.vector - f2.vector).norm() < 1.0e-12


def test_interpolation_function(mesh):
    V = FunctionSpace(mesh, ("Lagrange", 1))
    u = Function(V)
    u.vector.set(1)
    Vh = FunctionSpace(mesh, ("Lagrange", 1))
    uh = Function(Vh)
    uh.interpolate(u)
    assert np.allclose(uh.vector.array, 1)
