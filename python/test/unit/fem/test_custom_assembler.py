# Copyright (C) 2019-2020 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tests for custom Python assemblers"""

import ctypes
import ctypes.util
import importlib
import math
import os
import pathlib
import time

import cffi
import numba
import numba.core.typing.cffi_utils as cffi_support
import numpy as np
import numpy.typing
import pytest

import dolfinx
import dolfinx.pkgconfig
import ufl
from dolfinx.fem import Function, FunctionSpace, form, transpose_dofmap
from dolfinx.fem.petsc import assemble_matrix, load_petsc_lib
from dolfinx.mesh import create_unit_square
from ufl import dx, inner

import petsc4py.lib
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py import get_config as PETSc_get_config

# Get details of PETSc install
petsc_dir = PETSc_get_config()['PETSC_DIR']
petsc_arch = petsc4py.lib.getPathArchPETSc()[1]

# Get PETSc int and scalar types
if np.dtype(PETSc.ScalarType).kind == 'c':
    complex = True
else:
    complex = False

scalar_size = np.dtype(PETSc.ScalarType).itemsize
index_size = np.dtype(PETSc.IntType).itemsize

if index_size == 8:
    c_int_t = "int64_t"
    ctypes_index: numpy.typing.DTypeLike = ctypes.c_int64
elif index_size == 4:
    c_int_t = "int32_t"
    ctypes_index = ctypes.c_int32
else:
    raise RuntimeError(f"Cannot translate PETSc index size into a C type, index_size: {index_size}.")

if complex and scalar_size == 16:
    c_scalar_t = "double _Complex"
    numba_scalar_t = numba.types.complex128
elif complex and scalar_size == 8:
    c_scalar_t = "float _Complex"
    numba_scalar_t = numba.types.complex64
elif not complex and scalar_size == 8:
    c_scalar_t = "double"
    numba_scalar_t = numba.types.float64
elif not complex and scalar_size == 4:
    c_scalar_t = "float"
    numba_scalar_t = numba.types.float32
else:
    raise RuntimeError(
        f"Cannot translate PETSc scalar type to a C type, complex: {complex} size: {scalar_size}.")


petsc_lib_ctypes = load_petsc_lib(ctypes.cdll.LoadLibrary)
# Get the PETSc MatSetValuesLocal function via ctypes
# ctypes does not support static types well, ignore type check errors
MatSetValues_ctypes = petsc_lib_ctypes.MatSetValuesLocal
MatSetValues_ctypes.argtypes = [ctypes.c_void_p, ctypes_index, ctypes.POINTER(  # type: ignore
    ctypes_index), ctypes_index, ctypes.POINTER(ctypes_index), ctypes.c_void_p, ctypes.c_int]  # type: ignore
del petsc_lib_ctypes


# CFFI - register complex types
ffi = cffi.FFI()
cffi_support.register_type(ffi.typeof('double _Complex'), numba.types.complex128)
cffi_support.register_type(ffi.typeof('float _Complex'), numba.types.complex64)

# Get MatSetValuesLocal from PETSc available via cffi in ABI mode
ffi.cdef("""int MatSetValuesLocal(void* mat, {0} nrow, const {0}* irow,
                                  {0} ncol, const {0}* icol, const {1}* y, int addv);
""".format(c_int_t, c_scalar_t))


petsc_lib_cffi = load_petsc_lib(ffi.dlopen)
MatSetValues_abi = petsc_lib_cffi.MatSetValuesLocal


# @pytest.fixture
def get_matsetvalues_api():
    """Make MatSetValuesLocal from PETSc available via cffi in API mode"""
    if dolfinx.pkgconfig.exists("dolfinx"):
        dolfinx_pc = dolfinx.pkgconfig.parse("dolfinx")
    else:
        raise RuntimeError("Could not find DOLFINx pkgconfig file")

    worker = os.getenv('PYTEST_XDIST_WORKER', None)
    module_name = "_petsc_cffi_{}".format(worker)
    if MPI.COMM_WORLD.Get_rank() == 0:
        ffibuilder = cffi.FFI()
        ffibuilder.cdef("""
            typedef int... PetscInt;
            typedef ... PetscScalar;
            typedef int... InsertMode;
            int MatSetValuesLocal(void* mat, PetscInt nrow, const PetscInt* irow,
                                PetscInt ncol, const PetscInt* icol,
                                const PetscScalar* y, InsertMode addv);
        """)
        ffibuilder.set_source(module_name, """
            #include "petscmat.h"
        """,
                              libraries=['petsc'],
                              include_dirs=[os.path.join(petsc_dir, petsc_arch, 'include'),
                                            os.path.join(petsc_dir, 'include')] + dolfinx_pc["include_dirs"],
                              library_dirs=[os.path.join(petsc_dir, petsc_arch, 'lib')],
                              extra_compile_args=[])

        # Build module in same directory as test file
        path = pathlib.Path(__file__).parent.absolute()
        ffibuilder.compile(tmpdir=path, verbose=True)

    MPI.COMM_WORLD.Barrier()

    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise ImportError("Failed to find CFFI generated module")
    module = importlib.util.module_from_spec(spec)

    cffi_support.register_module(module)
    cffi_support.register_type(module.ffi.typeof("PetscScalar"), numba_scalar_t)
    return module.lib.MatSetValuesLocal


# See https://github.com/numba/numba/issues/4036 for why we need 'sink'
@numba.njit
def sink(*args):
    pass


@numba.njit(fastmath=True)
def area(x0, x1, x2) -> float:
    """Compute the area of a triangle embedded in 2D from the three vertices"""
    a = (x1[0] - x2[0])**2 + (x1[1] - x2[1])**2
    b = (x0[0] - x2[0])**2 + (x0[1] - x2[1])**2
    c = (x0[0] - x1[0])**2 + (x0[1] - x1[1])**2
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
def assemble_vector_ufc(b, kernel, mesh, dofmap, num_cells):
    """Assemble provided FFCx/UFC kernel over a mesh into the array b"""
    v, x = mesh
    entity_local_index = np.array([0], dtype=np.intc)
    perm = np.array([0], dtype=np.uint8)
    geometry = np.zeros((3, 3))
    coeffs = np.zeros(1, dtype=PETSc.ScalarType)
    constants = np.zeros(1, dtype=PETSc.ScalarType)

    b_local = np.zeros(3, dtype=PETSc.ScalarType)
    for cell in range(num_cells):
        # FIXME: This assumes a particular geometry dof layout
        for j in range(3):
            geometry[j] = x[v[cell, j], :]
        b_local.fill(0.0)
        kernel(ffi.from_buffer(b_local), ffi.from_buffer(coeffs),
               ffi.from_buffer(constants),
               ffi.from_buffer(geometry), ffi.from_buffer(entity_local_index),
               ffi.from_buffer(perm))
        for j in range(3):
            b[dofmap[cell, j]] += b_local[j]


@numba.njit(fastmath=True)
def assemble_matrix_cffi(A, mesh, dofmap, num_cells, set_vals, mode):
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
        set_vals(A, 3, ffi.from_buffer(pos), 3, ffi.from_buffer(pos), ffi.from_buffer(A_local), mode)
    sink(A_local, dofmap)


@numba.njit
def assemble_matrix_ctypes(A, mesh, dofmap, num_cells, set_vals, mode):
    """Assemble P1 mass matrix over a mesh into the PETSc matrix A"""
    v, x = mesh
    q = np.array([[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]], dtype=np.double)
    weights = np.full(3, 1.0 / 3.0, dtype=np.double)

    # Loop over cells
    N = np.empty(3, dtype=np.double)
    A_local = np.empty((3, 3), dtype=PETSc.ScalarType)
    for cell in range(num_cells):
        # FIXME: This assumes a particular geometry dof layout
        cell_area = area(x[v[cell, 0]], x[v[cell, 1]], x[v[cell, 2]])

        # Loop over quadrature points
        A_local[:] = 0.0
        for j in range(q.shape[0]):
            N[0], N[1], N[2] = 1.0 - q[j, 0] - q[j, 1], q[j, 0], q[j, 1]
            for row in range(3):
                for col in range(3):
                    A_local[row, col] += weights[j] * cell_area * N[row] * N[col]

        rows = cols = dofmap[cell, :]
        set_vals(A, 3, rows.ctypes, 3, cols.ctypes, A_local.ctypes, mode)


def test_custom_mesh_loop_rank1():

    # Create mesh and function space
    mesh = create_unit_square(MPI.COMM_WORLD, 64, 64)
    V = FunctionSpace(mesh, ("Lagrange", 1))

    # Unpack mesh and dofmap data
    num_owned_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    num_cells = num_owned_cells + mesh.topology.index_map(mesh.topology.dim).num_ghosts
    x_dofs = mesh.geometry.dofmap.array.reshape(num_cells, 3)
    x = mesh.geometry.x
    dofmap = V.dofmap.list.array.reshape(num_cells, 3)
    dofmap_t = transpose_dofmap(V.dofmap.list, num_owned_cells)

    # Assemble with pure Numba function (two passes, first will include
    # JIT overhead)
    b0 = Function(V)
    for i in range(2):
        b = b0.x.array
        b[:] = 0.0
        start = time.time()
        assemble_vector(b, (x_dofs, x), dofmap, num_owned_cells)
        end = time.time()
        print("Time (numba, pass {}): {}".format(i, end - start))
    b0.vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert b0.vector.sum() == pytest.approx(1.0)

    # Assemble with pure Numba function using parallel loop (two passes,
    # first will include JIT overhead)
    btmp = Function(V)
    for i in range(2):
        b = btmp.x.array
        b[:] = 0.0
        start = time.time()
        assemble_vector_parallel(b, x_dofs, x, dofmap_t.array, dofmap_t.offsets, num_owned_cells)
        end = time.time()
        print("Time (numba parallel, pass {}): {}".format(i, end - start))
    btmp.vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert (btmp.vector - b0.vector).norm() == pytest.approx(0.0)

    # Test against generated code and general assembler
    v = ufl.TestFunction(V)
    L = inner(1.0, v) * dx
    Lf = form(L)
    start = time.time()
    b1 = dolfinx.fem.petsc.assemble_vector(Lf)
    end = time.time()
    print("Time (C++, pass 0):", end - start)

    with b1.localForm() as b_local:
        b_local.set(0.0)
    start = time.time()
    dolfinx.fem.petsc.assemble_vector(b1, Lf)
    end = time.time()
    print("Time (C++, pass 1):", end - start)
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert (b1 - b0.vector).norm() == pytest.approx(0.0)

    # Assemble using generated tabulate_tensor kernel and Numba assembler
    ffcxtype = "double _Complex" if np.issubdtype(PETSc.ScalarType, np.complexfloating) else "double"
    b3 = Function(V)
    ufcx_form, module, code = dolfinx.jit.ffcx_jit(
        mesh.comm, L, form_compiler_options={"scalar_type": ffcxtype})

    nptype = "complex128" if np.issubdtype(PETSc.ScalarType, np.complexfloating) else "float64"
    # First 0 for "cell" integrals, second 0 for the first one, i.e. default domain
    kernel = getattr(ufcx_form.integrals(0)[0], f"tabulate_tensor_{nptype}")

    for i in range(2):
        b = b3.x.array
        b[:] = 0.0
        start = time.time()
        assemble_vector_ufc(b, kernel, (x_dofs, x), dofmap, num_owned_cells)
        end = time.time()
        print("Time (numba/cffi, pass {}): {}".format(i, end - start))
    b3.vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert (b3.vector - b0.vector).norm() == pytest.approx(0.0)


def test_custom_mesh_loop_ctypes_rank2():
    """Test numba assembler for bilinear form"""

    # Create mesh and function space
    mesh = create_unit_square(MPI.COMM_WORLD, 64, 64)
    V = FunctionSpace(mesh, ("Lagrange", 1))

    # Extract mesh and dofmap data
    num_owned_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    num_cells = num_owned_cells + mesh.topology.index_map(mesh.topology.dim).num_ghosts
    x_dofs = mesh.geometry.dofmap.array.reshape(num_cells, 3)
    x = mesh.geometry.x
    dofmap = V.dofmap.list.array.reshape(num_cells, 3).astype(np.dtype(PETSc.IntType))

    # Generated case with general assembler
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = form(inner(u, v) * dx)
    A0 = assemble_matrix(a)
    A0.assemble()
    A0.zeroEntries()

    start = time.time()
    dolfinx.fem.petsc.assemble_matrix(A0, a)
    end = time.time()
    print("Time (C++, pass 2):", end - start)
    A0.assemble()

    # Custom case
    A1 = A0.copy()
    for i in range(2):
        A1.zeroEntries()
        mat = A1.handle
        start = time.time()
        assemble_matrix_ctypes(mat, (x_dofs, x), dofmap, num_owned_cells,
                               MatSetValues_ctypes, PETSc.InsertMode.ADD_VALUES)
        end = time.time()
        print("Time (numba, pass {}): {}".format(i, end - start))
        A1.assemble()

    assert (A0 - A1).norm() == pytest.approx(0.0, abs=1.0e-9)


@pytest.mark.parametrize("set_vals", [MatSetValues_abi, get_matsetvalues_api()])
def test_custom_mesh_loop_cffi_rank2(set_vals):
    """Test numba assembler for bilinear form"""

    mesh = create_unit_square(MPI.COMM_WORLD, 64, 64)
    V = FunctionSpace(mesh, ("Lagrange", 1))

    # Test against generated code and general assembler
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = form(inner(u, v) * dx)
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
    num_cells = num_owned_cells + mesh.topology.index_map(mesh.topology.dim).num_ghosts
    x_dofs = mesh.geometry.dofmap.array.reshape(num_cells, 3)
    x = mesh.geometry.x
    dofmap = V.dofmap.list.array.reshape(num_cells, 3).astype(np.dtype(PETSc.IntType))

    A1 = A0.copy()
    for i in range(2):
        A1.zeroEntries()
        start = time.time()
        assemble_matrix_cffi(A1.handle, (x_dofs, x), dofmap, num_owned_cells, set_vals, PETSc.InsertMode.ADD_VALUES)
        end = time.time()
        print("Time (Numba, pass {}): {}".format(i, end - start))
        A1.assemble()

    assert (A1 - A0).norm() == pytest.approx(0.0)
