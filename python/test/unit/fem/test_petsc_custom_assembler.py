# Copyright (C) 2019-2024 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tests for custom Python assemblers."""

import math
import time

from mpi4py import MPI

try:
    from petsc4py import PETSc

    from dolfinx.fem.petsc import assemble_matrix
except ImportError:
    pass

try:
    import numba

except ImportError:
    pass

import numpy as np
import pytest

import ufl
from dolfinx.fem import form, functionspace
from dolfinx.fem.petsc import cffi_utils as petsc_cffi
from dolfinx.fem.petsc import ctypes_utils as petsc_ctypes
from dolfinx.fem.petsc import numba_utils as petsc_numba
from dolfinx.mesh import create_unit_square

pytest.importorskip("petsc4py")
cffi = pytest.importorskip("cffi")
cffi_support = pytest.importorskip("numba.core.typing.cffi_utils")
numba = pytest.importorskip("numba")

# Get PETSc MatSetValuesLocal interfaces
try:
    MatSetValuesLocal = petsc_numba.MatSetValuesLocal
    MatSetValuesLocal_ctypes = petsc_ctypes.MatSetValuesLocal
    MatSetValuesLocal_abi = petsc_cffi.MatSetValuesLocal
except AttributeError:
    MatSetValuesLocal_abi = None

ffi = cffi.FFI()


@numba.njit
def set_vals_numba(A, m, rows, n, cols, data, mode):
    MatSetValuesLocal(A, 3, rows.ctypes, 3, cols.ctypes, data.ctypes, mode)


@numba.njit
def set_vals_cffi(A, m, rows, n, cols, data, mode):
    MatSetValuesLocal_abi(
        A, m, ffi.from_buffer(rows), n, ffi.from_buffer(cols), ffi.from_buffer(data), mode
    )


@numba.njit
def set_vals_ctypes(A, m, rows, n, cols, data, mode):
    MatSetValuesLocal_ctypes(A, m, rows.ctypes, n, cols.ctypes, data.ctypes, mode)


# See https://github.com/numba/numba/issues/4036 for why we need 'sink'
@numba.njit
def sink(*args):
    pass


@numba.njit(fastmath=True)
def area(x0, x1, x2) -> float:
    """Compute the area of a triangle embedded in 2D from the three vertices"""
    a = (x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2
    b = (x0[0] - x2[0]) ** 2 + (x0[1] - x2[1]) ** 2
    c = (x0[0] - x1[0]) ** 2 + (x0[1] - x1[1]) ** 2
    return math.sqrt(2 * (a * b + a * c + b * c) - (a**2 + b**2 + c**2)) / 4.0


@numba.njit(fastmath=True)
def assemble_petsc_matrix(A, mesh, dofmap, num_cells, set_vals, mode):
    """Assemble P1 mass matrix over a mesh into the PETSc matrix A"""
    # Mesh data
    v, x = mesh

    # Quadrature points and weights
    q = np.array([[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]], dtype=np.double)
    weights = np.full(3, 1.0 / 3.0, dtype=np.double)

    # Loop over cells
    N = np.empty(3, dtype=np.double)
    A_local = np.empty((3, 3), dtype=PETSc.ScalarType)
    for cell in range(num_cells):
        cell_area = area(x[v[cell, 0]], x[v[cell, 1]], x[v[cell, 2]])

        # Loop over quadrature points
        A_local[:] = 0.0
        for j in range(q.shape[0]):
            N[0], N[1], N[2] = 1.0 - q[j, 0] - q[j, 1], q[j, 0], q[j, 1]
            for row in range(3):
                for col in range(3):
                    A_local[row, col] += weights[j] * cell_area * N[row] * N[col]

        # Add to global tensor
        pos = dofmap[cell, :]
        set_vals(A, 3, pos, 3, pos, A_local, mode)
    sink(A_local, dofmap)


@pytest.mark.skipif(cffi.__version_info__ == (1, 17, 1), reason="bug in cffi 1.17.1 for complex")
@pytest.mark.petsc4py
@pytest.mark.parametrize(
    "set_vals,backend",
    [
        (set_vals_numba, "numba"),
        (set_vals_ctypes, "ctypes"),
        (set_vals_cffi, "cffi_abi"),
    ],
)
def test_custom_mesh_loop_petsc_rank2(set_vals, backend):
    """Test numba assembler for a bilinear form."""

    mesh = create_unit_square(MPI.COMM_WORLD, 64, 64)
    V = functionspace(mesh, ("Lagrange", 1))

    # Test against generated code and general assembler
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = form(ufl.inner(u, v) * ufl.dx)
    A0 = assemble_matrix(a)
    A0.assemble()

    A0.zeroEntries()
    start = time.time()
    assemble_matrix(A0, a)
    end = time.time()
    print("Time (C++, pass 2):", end - start)
    A0.assemble()

    # Unpack mesh and dofmap data
    num_owned_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    x_dofs = mesh.geometry.dofmap
    x = mesh.geometry.x
    dofmap = V.dofmap.list.astype(np.dtype(PETSc.IntType))

    A1 = A0.copy()
    for i in range(2):
        A1.zeroEntries()
        start = time.time()
        assemble_petsc_matrix(
            A1.handle, (x_dofs, x), dofmap, num_owned_cells, set_vals, PETSc.InsertMode.ADD_VALUES
        )
        end = time.time()
        print(f"Time (Numba/{backend}, pass {i}): {end - start}")
        A1.assemble()
    assert (A1 - A0).norm() == pytest.approx(0.0, abs=1.0e-9)

    A0.destroy()
    A1.destroy()
