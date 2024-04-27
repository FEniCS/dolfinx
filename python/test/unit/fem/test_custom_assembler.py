# Copyright (C) 2019-2024 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tests for custom Python assemblers."""

import importlib
import math
import os
import pathlib
import time

from mpi4py import MPI

import numpy as np
import pytest

import dolfinx
import dolfinx.pkgconfig
import ufl
from dolfinx.fem import Function, form, functionspace
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.mesh import create_unit_square
from dolfinx.utils import cffi_utils as petsc_cffi
from dolfinx.utils import ctypes_utils as petsc_ctypes
from dolfinx.utils import numba_utils as petsc_numba

cffi = pytest.importorskip("cffi")
cffi_support = pytest.importorskip("numba.core.typing.cffi_utils")
numba = pytest.importorskip("numba")

# Get PETSc MatSetValuesLocal interfaces
MatSetValuesLocal = petsc_numba.MatSetValuesLocal
MatSetValuesLocal_ctypes = petsc_ctypes.MatSetValuesLocal
try:
    MatSetValuesLocal_abi = petsc_cffi.MatSetValuesLocal
except AttributeError:
    MatSetValuesLocal_abi = None


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


ffi = cffi.FFI()


def get_matsetvalues_cffi_api():
    """Make MatSetValuesLocal from PETSc available via cffi in API mode.

    This function is not (yet) in the DOLFINx module because it is complicated
    by needing to compile code.
    """
    if dolfinx.pkgconfig.exists("dolfinx"):
        dolfinx_pc = dolfinx.pkgconfig.parse("dolfinx")
    else:
        raise RuntimeError("Could not find DOLFINx pkg-config file")

    import petsc4py.lib
    from petsc4py import get_config as PETSc_get_config

    cffi_support.register_type(ffi.typeof("float _Complex"), numba.types.complex64)
    cffi_support.register_type(ffi.typeof("double _Complex"), numba.types.complex128)

    petsc_dir = PETSc_get_config()["PETSC_DIR"]
    petsc_arch = petsc4py.lib.getPathArchPETSc()[1]

    worker = os.getenv("PYTEST_XDIST_WORKER", None)
    module_name = f"_petsc_cffi_{worker}"
    if MPI.COMM_WORLD.Get_rank() == 0:
        ffibuilder = cffi.FFI()
        ffibuilder.cdef(
            """typedef int... PetscInt;
                           typedef ... PetscScalar;
                           typedef int... InsertMode;
                           int MatSetValuesLocal(void* mat, PetscInt nrow, const PetscInt* irow,
                                PetscInt ncol, const PetscInt* icol,
                                const PetscScalar* y, InsertMode addv);"""
        )
        ffibuilder.set_source(
            module_name,
            '#include "petscmat.h"',
            libraries=["petsc"],
            include_dirs=[
                os.path.join(petsc_dir, petsc_arch, "include"),
                os.path.join(petsc_dir, "include"),
            ]
            + dolfinx_pc["include_dirs"],
            library_dirs=[os.path.join(petsc_dir, petsc_arch, "lib")],
            extra_compile_args=[],
        )

        # Build module in same directory as test file
        path = pathlib.Path(__file__).parent.absolute()
        ffibuilder.compile(tmpdir=path, verbose=True)

    MPI.COMM_WORLD.Barrier()
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise ImportError("Failed to find CFFI generated module")
    module = importlib.util.module_from_spec(spec)
    cffi_support.register_module(module)
    cffi_support.register_type(module.ffi.typeof("PetscScalar"), petsc_numba._scalar)
    return module.lib.MatSetValuesLocal


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
        )
        for j in range(3):
            b[dofmap[cell, j]] += b_local[j]


@numba.njit(fastmath=True)
def assemble_petsc_matrix(A, mesh, dofmap, num_cells, set_vals, mode):
    """Assemble P1 mass matrix over a mesh into the PETSc matrix A"""
    from petsc4py import PETSc

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


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
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
    # btmp.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    # assert (btmp.x.petsc_vec - b0.x.petsc_vec).norm() == pytest.approx(0.0)

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
    ufcx_form, module, code = dolfinx.jit.ffcx_jit(
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
    from petsc4py import PETSc

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
