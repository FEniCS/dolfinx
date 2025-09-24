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
    import numba

    from ffcx.codegeneration.utils import get_void_pointer
except ImportError:
    pass

import numpy as np
import pytest

import dolfinx
import ufl
from dolfinx.fem import Function, form, functionspace
from dolfinx.mesh import create_unit_square

cffi = pytest.importorskip("cffi")
cffi_support = pytest.importorskip("numba.core.typing.cffi_utils")
numba = pytest.importorskip("numba")

ffi = cffi.FFI()


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
def assemble_vector(b, mesh, dofmap, num_cells):
    """Assemble simple linear form over a mesh into the array b"""
    v, x = mesh
    q0, q1 = 1 / 3.0, 1 / 3.0
    for cell in range(num_cells):
        # FIXME: This assumes a particular geometry dof layout
        A = area(x[v[cell, 0]], x[v[cell, 1]], x[v[cell, 2]])
        b[dofmap[cell, 0]] += A * (1.0 - q0 - q1)
        b[dofmap[cell, 1]] += A * q0
        b[dofmap[cell, 2]] += A * q1


@numba.njit(parallel=True, fastmath=True)
def assemble_vector_parallel(b, v, x, dofmap_t_data, dofmap_t_offsets, num_cells):
    """Assemble simple linear form over a mesh into the array b using a parallel loop"""
    q0 = 1 / 3.0
    q1 = 1 / 3.0
    b_unassembled = np.empty((num_cells, 3), dtype=b.dtype)
    for cell in numba.prange(num_cells):
        # FIXME: This assumes a particular geometry dof layout
        A = area(x[v[cell, 0]], x[v[cell, 1]], x[v[cell, 2]])
        b_unassembled[cell, 0] = A * (1.0 - q0 - q1)
        b_unassembled[cell, 1] = A * q0
        b_unassembled[cell, 2] = A * q1

    # Accumulate values in RHS
    _b_unassembled = b_unassembled.reshape(num_cells * 3)
    for index in numba.prange(dofmap_t_offsets.shape[0] - 1):
        for p in range(dofmap_t_offsets[index], dofmap_t_offsets[index + 1]):
            b[index] += _b_unassembled[dofmap_t_data[p]]


@numba.njit(fastmath=True)
def assemble_vector_ufc(b, kernel, mesh, dofmap, num_cells, dtype):
    """Assemble provided FFCx/UFC kernel over a mesh into the array b"""
    v, x = mesh
    entity_local_index = np.array([0], dtype=np.intc)
    perm = np.array([0], dtype=np.uint8)
    geometry = np.zeros((3, 3), dtype=x.dtype)
    coeffs = np.zeros(1, dtype=dtype)
    constants = np.zeros(1, dtype=dtype)
    custom_data = np.zeros(1, dtype=np.int64)
    custom_data_ptr = get_void_pointer(custom_data)

    b_local = np.zeros(3, dtype=dtype)
    for cell in range(num_cells):
        # FIXME: This assumes a particular geometry dof layout
        for j in range(3):
            geometry[j] = x[v[cell, j], :]
        b_local.fill(0.0)
        kernel(
            ffi.from_buffer(b_local),
            ffi.from_buffer(coeffs),
            ffi.from_buffer(constants),
            ffi.from_buffer(geometry),
            ffi.from_buffer(entity_local_index),
            ffi.from_buffer(perm),
            custom_data_ptr,
        )
        for j in range(3):
            b[dofmap[cell, j]] += b_local[j]


@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        pytest.param(
            np.complex64,
            marks=[
                pytest.mark.xfail_win32_complex,
                pytest.mark.skipif(
                    cffi.__version_info__ > (1, 16, 99) and cffi.__version_info__ <= (2, 0, 0),
                    reason="bug in cffi 1.17.0/1 and 2.0.0 for complex",
                ),
            ],
        ),
        pytest.param(
            np.complex128,
            marks=[
                pytest.mark.xfail_win32_complex,
                pytest.mark.skipif(
                    cffi.__version_info__ > (1, 16, 99) and cffi.__version_info__ <= (2, 0, 0),
                    reason="bug in cffi 1.17.0/1 and 2.0.0 for complex",
                ),
            ],
        ),
    ],
)
def test_custom_mesh_loop_rank1(dtype):
    mesh = create_unit_square(MPI.COMM_WORLD, 64, 64, dtype=dtype(0).real.dtype)
    V = functionspace(mesh, ("Lagrange", 1))

    # Unpack mesh and dofmap data
    num_owned_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    x_dofs = mesh.geometry.dofmap
    x = mesh.geometry.x
    dofmap = V.dofmap.list

    # Assemble with pure Numba function (two passes, first will include
    # JIT overhead)
    b0 = Function(V, dtype=dtype)
    for i in range(2):
        b = b0.x.array
        b[:] = 0.0
        start = time.time()
        assemble_vector(b, (x_dofs, x), dofmap, num_owned_cells)
        end = time.time()
        print(f"Time (numba, pass {i}): {end - start}")
    b0.x.scatter_reverse(dolfinx.la.InsertMode.add)
    b0sum = np.sum(b0.x.array[: b0.x.index_map.size_local * b0.x.block_size])
    assert mesh.comm.allreduce(b0sum, op=MPI.SUM) == pytest.approx(1.0)

    # NOTE: Parallel (threaded) Numba can cause problems with MPI
    # Assemble with pure Numba function using parallel loop (two passes,
    # first will include JIT overhead)
    # from dolfinx.fem import transpose_dofmap
    # dofmap_t = transpose_dofmap(V.dofmap.list, num_owned_cells)
    # btmp = Function(V)
    # for i in range(2):
    #     b = btmp.x.array
    #     b[:] = 0.0
    #     start = time.time()
    #     assemble_vector_parallel(b, x_dofs, x, dofmap_t.array, dofmap_t.offsets, num_owned_cells)
    #     end = time.time()
    #     print("Time (numba parallel, pass {}): {}".format(i, end - start))
    # btmp.x.scatter_reverse(dolfinx.la.InsertMode.add)
    # assert (btmp.x.array - b0.x.array).norm() == pytest.approx(0.0)

    # Test against generated code and general assembler
    v = ufl.TestFunction(V)
    L = ufl.inner(1.0, v) * ufl.dx
    Lf = form(L, dtype=dtype)
    start = time.time()
    b1 = dolfinx.fem.assemble_vector(Lf)
    end = time.time()
    print("Time (C++, pass 0):", end - start)

    b1.array[:] = 0
    start = time.time()
    dolfinx.fem.assemble_vector(b1.array, Lf)
    end = time.time()
    print("Time (C++, pass 1):", end - start)
    b1.scatter_reverse(dolfinx.la.InsertMode.add)
    assert np.linalg.norm(b1.array - b0.x.array) == pytest.approx(0.0, abs=1.0e-8)

    # Assemble using generated tabulate_tensor kernel and Numba
    # assembler
    b3 = Function(V, dtype=dtype)
    ufcx_form, _module, _code = dolfinx.jit.ffcx_jit(
        mesh.comm, L, form_compiler_options={"scalar_type": dtype}
    )

    # Get the one and only kernel
    kernel = getattr(ufcx_form.form_integrals[0], f"tabulate_tensor_{np.dtype(dtype).name}")

    for i in range(2):
        b = b3.x.array
        b[:] = 0.0
        start = time.time()
        assemble_vector_ufc(b, kernel, (x_dofs, x), dofmap, num_owned_cells, dtype)
        end = time.time()
        print(f"Time (numba/cffi, pass {i}): {end - start}")
    b3.x.scatter_reverse(dolfinx.la.InsertMode.add)
    assert np.linalg.norm(b3.x.array - b0.x.array) == pytest.approx(0.0, abs=1e-8)
