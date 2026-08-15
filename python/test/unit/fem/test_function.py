# Copyright (C) 2011-2022 Garth N. Wells, Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the Function class."""

import importlib
import sys

from mpi4py import MPI

import cffi
import numpy as np
import pytest

import ufl
from basix.ufl import element, mixed_element
from dolfinx import la
from dolfinx.fem import Function, functionspace
from dolfinx.geometry import bb_tree, compute_colliding_cells, compute_collisions_points
from dolfinx.mesh import create_mesh, create_unit_cube


@pytest.fixture
def mesh():
    return create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)


@pytest.fixture
def V(mesh):
    return functionspace(mesh, ("Lagrange", 1))


@pytest.fixture
def W(mesh):
    gdim = mesh.geometry.dim
    return functionspace(mesh, ("Lagrange", 1, (gdim,)))


@pytest.fixture
def Q(mesh):
    gdim = mesh.geometry.dim
    return functionspace(mesh, ("Lagrange", 1, (gdim, gdim)))


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


@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        pytest.param(
            np.complex64,
            marks=pytest.mark.xfail(
                sys.platform.startswith("win32"),
                raises=NotImplementedError,
                reason="missing _Complex",
            ),
        ),
        np.float64,
        pytest.param(
            np.complex128,
            marks=pytest.mark.xfail(
                sys.platform.startswith("win32"),
                raises=NotImplementedError,
                reason="missing _Complex",
            ),
        ),
    ],
)
def test_eval(dtype):
    xdtype = dtype(0).real.dtype
    mesh = create_unit_cube(MPI.COMM_WORLD, 3, 3, 3, dtype=xdtype)
    gdim = mesh.geometry.dim
    V = functionspace(mesh, ("Lagrange", 1))
    W = functionspace(mesh, ("Lagrange", 1, (gdim,)))
    Q = functionspace(mesh, ("Lagrange", 1, (gdim, gdim)))

    u1 = Function(V, dtype=dtype)
    u2 = Function(W, dtype=dtype)
    u3 = Function(Q, dtype=dtype)

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
    tree = bb_tree(mesh, mesh.topology.dim, padding=0.0)
    cell_candidates = compute_collisions_points(tree, x0)
    cell = compute_colliding_cells(mesh, cell_candidates, x0).array
    assert len(cell) > 0
    first_cell = cell[0]
    # The same expression through two spaces: agrees to a few ulp.
    tol = max(1.0e-15, 10 * np.finfo(dtype).eps)
    assert np.allclose(u3.eval(x0, first_cell)[:3], u2.eval(x0, first_cell), rtol=tol, atol=tol)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        pytest.param(
            np.complex64,
            marks=pytest.mark.xfail(
                sys.platform.startswith("win32"),
                raises=NotImplementedError,
                reason="missing _Complex",
            ),
        ),
        np.float64,
        pytest.param(
            np.complex128,
            marks=pytest.mark.xfail(
                sys.platform.startswith("win32"),
                raises=NotImplementedError,
                reason="missing _Complex",
            ),
        ),
    ],
)
def test_eval_manifold(dtype):
    xdtype = dtype(0).real.dtype
    # Simple two-triangle surface in 3d
    vertices = np.array(
        [(0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        dtype=xdtype,
    )
    cells = [(0, 1, 2), (0, 1, 3)]
    domain = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,), dtype=xdtype))
    mesh = create_mesh(MPI.COMM_WORLD, cells, domain, vertices)
    Q = functionspace(mesh, ("Lagrange", 1))
    u = Function(Q, dtype=dtype)
    u.interpolate(lambda x: x[0] + x[1])
    # Exact in real arithmetic; absorbs interpolation and evaluation rounding.
    rtol = max(1.0e-5, 1000 * np.finfo(dtype).eps)
    assert np.isclose(u.eval([0.75, 0.25, 0.5], 0)[0], 1.0, rtol=rtol)


def test_interpolation_mismatch_rank0(W):
    u = Function(W)
    with pytest.raises(RuntimeError):
        u.interpolate(lambda x: np.ones(x.shape[1]))


def test_interpolation_mismatch_rank1(W):
    u = Function(W)
    with pytest.raises(RuntimeError):
        u.interpolate(lambda x: np.ones((2, x.shape[1])))


@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        pytest.param(
            np.complex64,
            marks=pytest.mark.xfail(
                sys.platform.startswith("win32"),
                raises=NotImplementedError,
                reason="missing _Complex",
            ),
        ),
        np.float64,
        pytest.param(
            np.complex128,
            marks=pytest.mark.xfail(
                sys.platform.startswith("win32"),
                raises=NotImplementedError,
                reason="missing _Complex",
            ),
        ),
    ],
)
def test_mixed_element_interpolation(dtype):
    xdtype = dtype(0).real.dtype
    mesh = create_unit_cube(MPI.COMM_WORLD, 3, 3, 3, dtype=xdtype)
    el = element("Lagrange", mesh.basix_cell(), 1, dtype=xdtype)
    V = functionspace(mesh, mixed_element([el, el]))
    u = Function(V, dtype=dtype)
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
    assert (w.x.array[:] == 1.0).all()  # /NOSONAR

    num_vertices = V.mesh.topology.index_map(0).size_global
    assert np.isclose(la.norm(w.x, la.Norm.l1) - num_vertices, 0)

    f.t = 2.0
    w.interpolate(f.eval)
    assert (w.x.array[:] == 2.0).all()  # /NOSONAR


def test_interpolation_rank1(W):
    def f(x):
        values = np.empty((3, x.shape[1]))
        values[0] = 1.0
        values[1] = 2.0
        values[2] = 3.0
        return values

    w = Function(W)
    w.interpolate(f)
    x = w.x.array
    assert x.max() == 3.0  # /NOSONAR
    assert x.min() == 1.0  # /NOSONAR

    num_vertices = W.mesh.topology.index_map(0).size_global
    assert round(la.norm(w.x, la.Norm.l1) - 6 * num_vertices, 7) == 0


@pytest.mark.parametrize("dtype,cdtype", [(np.float32, "float"), (np.float64, "double")])
def test_cffi_expression(dtype, cdtype):
    mesh = create_unit_cube(MPI.COMM_WORLD, 3, 3, 3, dtype=dtype)
    V = functionspace(mesh, ("Lagrange", 1))

    code_h = f"void eval({cdtype}* values, int num_points, int value_size, const {cdtype}* x);"
    code_c = """
        void eval(xtype* values, int num_points, int value_size, const xtype* x)
        {
        /* x0 + x1 */
        for (int i = 0; i < num_points; ++i)
          values[i] = x[i] + x[i + num_points];
        }
    """
    code_c = code_c.replace("xtype", cdtype)

    # Build the kernel
    module = "_expr_eval" + cdtype + str(MPI.COMM_WORLD.rank)
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
    f1 = Function(V, dtype=dtype)
    f1.interpolate(int(eval_ptr))

    f2 = Function(V, dtype=dtype)
    f2.interpolate(lambda x: x[0] + x[1])

    f1.x.array[:] -= f2.x.array
    assert la.norm(f1.x) < 1.0e-12


@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        pytest.param(
            np.complex64,
            marks=pytest.mark.xfail(
                sys.platform.startswith("win32"),
                raises=NotImplementedError,
                reason="missing _Complex",
            ),
        ),
        np.float64,
        pytest.param(
            np.complex128,
            marks=pytest.mark.xfail(
                sys.platform.startswith("win32"),
                raises=NotImplementedError,
                reason="missing _Complex",
            ),
        ),
    ],
)
def test_interpolation_function(dtype):
    xdtype = dtype(0).real.dtype
    mesh = create_unit_cube(MPI.COMM_WORLD, 3, 3, 3, dtype=xdtype)
    V = functionspace(mesh, ("Lagrange", 1))
    u = Function(V, dtype=dtype)
    u.x.array[:] = 1
    Vh = functionspace(mesh, ("Lagrange", 1))
    uh = Function(Vh, dtype=dtype)
    uh.interpolate(u)
    # Exact in real arithmetic; absorbs the dual evaluation rounding.
    assert np.allclose(uh.x.array, 1, rtol=max(1.0e-5, 100 * np.finfo(dtype).eps))
