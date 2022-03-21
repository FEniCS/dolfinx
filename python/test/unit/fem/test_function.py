# Copyright (C) 2011-2021 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the Function class"""

import importlib

import cffi
import numpy as np
import pytest

import ufl
from dolfinx.fem import (Function, FunctionSpace, TensorFunctionSpace,
                         VectorFunctionSpace)
from dolfinx.geometry import (BoundingBoxTree, compute_colliding_cells,
                              compute_collisions)
from dolfinx.mesh import create_mesh, create_unit_cube

from mpi4py import MPI
from petsc4py import PETSc


@pytest.fixture
def mesh():
    return create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)


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
    assert u.name == "f"
    assert v.name == "v"
    assert str(v) == "v"


def test_copy(V):
    u = Function(V)
    u.interpolate(lambda x: x[0] + 2 * x[1])
    v = u.copy()
    assert np.allclose(u.x.array, v.x.array)
    u.x.array[:] = 1
    assert not np.allclose(u.x.array, v.x.array)


def test_eval(V, W, Q, mesh):
    u1 = Function(V)
    u2 = Function(W)
    u3 = Function(Q)

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

    u1.interpolate(lambda x: x[0] + x[1] + x[2])
    u2.interpolate(e2)
    u3.interpolate(e3)

    x0 = (mesh.geometry.x[0] + mesh.geometry.x[1]) / 2.0
    tree = BoundingBoxTree(mesh, mesh.geometry.dim)
    cell_candidates = compute_collisions(tree, x0)
    cell = compute_colliding_cells(mesh, cell_candidates, x0)
    first_cell = cell[0]
    assert np.allclose(u3.eval(x0, first_cell)[:3], u2.eval(x0, first_cell), rtol=1e-15, atol=1e-15)


@pytest.mark.skip_in_parallel
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
    u = Function(W)
    with pytest.raises(RuntimeError):
        u.interpolate(lambda x: np.ones(x.shape[1]))


def test_interpolation_mismatch_rank1(W):
    u = Function(W)
    with pytest.raises(RuntimeError):
        u.interpolate(lambda x: np.ones((2, x.shape[1])))


def test_mixed_element_interpolation():
    mesh = create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)
    el = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, ufl.MixedElement([el, el]))
    u = Function(V)
    with pytest.raises(RuntimeError):
        u.interpolate(lambda x: np.ones(2, x.shape[1]))


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
    assert (w.x.array[:] == 1.0).all()

    num_vertices = V.mesh.topology.index_map(0).size_global
    assert np.isclose(w.vector.norm(PETSc.NormType.N1) - num_vertices, 0)

    f.t = 2.0
    w.interpolate(f.eval)
    assert (w.x.array[:] == 2.0).all()


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


def test_cffi_expression(V):
    code_h = """
    void eval(double* values, int num_points, int value_size, const double* x);
    """

    code_c = """
    void eval(double* values, int num_points, int value_size, const double* x)
    {
      /* x0 + x1 */
      for (int i = 0; i < num_points; ++i)
        values[i  + 0] = x[i] + x[i + num_points];
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
    f1 = Function(V, dtype=np.float64)
    f1.interpolate(int(eval_ptr))

    f2 = Function(V, dtype=np.float64)
    f2.interpolate(lambda x: x[0] + x[1])

    f1.x.array[:] -= f2.x.array
    assert f1.x.norm() < 1.0e-12


def test_interpolation_function(mesh):
    V = FunctionSpace(mesh, ("Lagrange", 1))
    u = Function(V)
    u.x.array[:] = 1
    Vh = FunctionSpace(mesh, ("Lagrange", 1))
    uh = Function(Vh)
    uh.interpolate(u)
    assert np.allclose(uh.x.array, 1)
