# Copyright (C) 2019 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tests for custom Python assemblers"""

import ctypes
import ctypes.util
import importlib
import math
import os
import time

import cffi
import numba
import numba.cffi_support
import numpy as np
import pytest
from petsc4py import PETSc

import dolfin
from ufl import dx, inner

petsc_dir = os.environ.get('PETSC_DIR', None)

# Get PETSc int and scalar types
if np.dtype(PETSc.ScalarType).kind == 'c':
    complex = True
else:
    complex = False

scalar_size = np.dtype(PETSc.ScalarType).itemsize
index_size = np.dtype(PETSc.IntType).itemsize

if index_size == 8:
    c_int_t = "int64_t"
    ctypes_index = ctypes.c_int64
elif index_size == 4:
    c_int_t = "int32_t"
    ctypes_index = ctypes.c_int32
else:
    raise RecursionError("Unknown PETSc index type.")

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
        "Cannot translate PETSc scalar type to a C type, complex: {} size: {}.".format(complex, scalar_size))


# Load PETSc library via ctypes
petsc_lib_name = ctypes.util.find_library("petsc")
if petsc_lib_name is not None:
    petsc_lib_ctypes = ctypes.CDLL(petsc_lib_name)
else:
    try:
        petsc_lib_ctypes = ctypes.CDLL(os.path.join(petsc_dir, "lib", "libpetsc.so"))
    except OSError:
        petsc_lib_ctypes = ctypes.CDLL(os.path.join(petsc_dir, "lib", "libpetsc.dylib"))
    except OSError:
        print("Could not load PETSc library for CFFI (ABI mode).")
        raise

# Get the PETSc MatSetValuesLocal function via ctypes
MatSetValues_ctypes = petsc_lib_ctypes.MatSetValuesLocal
MatSetValues_ctypes.argtypes = (ctypes.c_void_p, ctypes_index, ctypes.POINTER(
    ctypes_index), ctypes_index, ctypes.POINTER(ctypes_index), ctypes.c_void_p, ctypes.c_int)
del petsc_lib_ctypes


ADD_VALUES = PETSc.InsertMode.ADD_VALUES


# CFFI - register complex types
ffi = cffi.FFI()
numba.cffi_support.register_type(ffi.typeof('double _Complex'),
                                 numba.types.complex128)
numba.cffi_support.register_type(ffi.typeof('float _Complex'),
                                 numba.types.complex64)


# Get MatSetValuesLocal from PETSc available via cffi in ABI mode
ffi.cdef("""int MatSetValuesLocal(void* mat, {0} nrow, const {0}* irow,
                                  {0} ncol, const {0}* icol, const {1}* y, int addv);
""".format(c_int_t, c_scalar_t))

if petsc_lib_name is not None:
    petsc_lib_cffi = ffi.dlopen(petsc_lib_name)
else:
    try:
        petsc_lib_cffi = ffi.dlopen(os.path.join(petsc_dir, "lib", "libpetsc.so"))
    except OSError:
        petsc_lib_cffi = ffi.dlopen(os.path.join(petsc_dir, "lib", "libpetsc.dylib"))
    except OSError:
        print("Could not load PETSc library for CFFI (ABI mode).")
        raise
MatSetValues_abi = petsc_lib_cffi.MatSetValuesLocal

# Make MatSetValuesLocal from PETSc available via cffi in API mode
worker = os.getenv('PYTEST_XDIST_WORKER', None)
module_name = "_petsc_cffi_{}".format(worker)
if dolfin.MPI.comm_world.Get_rank() == 0:
    os.environ["CC"] = "mpicc"
    petsc_dir = os.environ.get('PETSC_DIR', None)
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
        # include "petscmat.h"
    """,
                          libraries=['petsc'],
                          include_dirs=[os.path.join(petsc_dir, 'include')],
                          library_dirs=[os.path.join(petsc_dir, 'lib')],
                          extra_compile_args=[])
    ffibuilder.compile(verbose=False)

dolfin.MPI.comm_world.barrier()

spec = importlib.util.find_spec(module_name)
if spec is None:
    raise ImportError("Failed to find CFFI generated module")
module = importlib.util.module_from_spec(spec)

numba.cffi_support.register_module(module)
MatSetValues_api = module.lib.MatSetValuesLocal
numba.cffi_support.register_type(module.ffi.typeof("PetscScalar"), numba_scalar_t)


# See https://github.com/numba/numba/issues/4036 for why we need 'sink'
@numba.njit
def sink(*args):
    pass


@numba.njit
def area(x0, x1, x2) -> float:
    """Compute the area of a triangle embedded in 2D from the three vertices"""
    a = (x1[0] - x2[0])**2 + (x1[1] - x2[1])**2
    b = (x0[0] - x2[0])**2 + (x0[1] - x2[1])**2
    c = (x0[0] - x1[0])**2 + (x0[1] - x1[1])**2
    return math.sqrt(2 * (a * b + a * c + b * c) - (a**2 + b**2 + c**2)) / 4.0


@numba.njit
def assemble_vector(b, mesh, x, dofmap):
    """Assemble simple linear form over a mesh into the array b"""
    connections, pos = mesh
    q0, q1 = 1 / 3.0, 1 / 3.0
    for i, cell in enumerate(pos[:-1]):
        num_vertices = pos[i + 1] - pos[i]
        c = connections[cell:cell + num_vertices]
        A = area(x[c[0]], x[c[1]], x[c[2]])
        b[dofmap[i * 3 + 0]] += A * (1.0 - q0 - q1)
        b[dofmap[i * 3 + 1]] += A * q0
        b[dofmap[i * 3 + 2]] += A * q1


@numba.njit
def assemble_vector_ufc(b, kernel, mesh, x, dofmap):
    """Assemble provided FFC/UFC kernel over a mesh into the array b"""
    connections, pos = mesh
    orientation = np.array([0], dtype=np.int32)
    geometry = np.zeros((3, 2))
    coeffs = np.zeros(1, dtype=PETSc.ScalarType)
    b_local = np.zeros(3, dtype=PETSc.ScalarType)
    for i, cell in enumerate(pos[:-1]):
        num_vertices = pos[i + 1] - pos[i]
        c = connections[cell:cell + num_vertices]
        for j in range(3):
            for k in range(2):
                geometry[j, k] = x[c[j], k]
        kernel(ffi.from_buffer(b_local), ffi.from_buffer(coeffs), ffi.from_buffer(geometry), ffi.from_buffer(orientation), ffi.from_buffer(orientation))
        for j in range(3):
            b[dofmap[i * 3 + j]] += b_local[j]


@numba.njit(fastmath=True)
def assemble_matrix_cffi(A, mesh, x, dofmap, set_vals, mode):
    """Assemble P1 mass matrix over a mesh into the PETSc matrix A"""

    # Mesh data
    connections, cell_ptr = mesh

    # Quadrature points and weights
    q = np.array([[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]], dtype=np.double)
    weights = np.full(3, 1.0 / 3.0, dtype=np.double)

    # Loop over cells
    N = np.empty(3, dtype=np.double)
    A_local = np.empty((3, 3), dtype=PETSc.ScalarType)
    for i, cell in enumerate(cell_ptr[:-1]):
        num_vertices = cell_ptr[i + 1] - cell_ptr[i]
        c = connections[cell:cell + num_vertices]
        cell_area = area(x[c[0]], x[c[1]], x[c[2]])

        # Loop over quadrature points
        A_local[:] = 0.0
        for j in range(q.shape[0]):
            N[0], N[1], N[2] = 1.0 - q[j, 0] - q[j, 1], q[j, 0], q[j, 1]
            for k in range(3):
                for l in range(3):
                    A_local[k, l] += weights[j] * cell_area * N[k] * N[l]

        # Add to global tensor
        pos = dofmap[3 * i:3 * i + 3]
        set_vals(A, 3, ffi.from_buffer(pos), 3, ffi.from_buffer(pos), ffi.from_buffer(A_local), mode)
    sink(A_local, dofmap)


@numba.njit
def assemble_matrix_ctypes(A, mesh, x, dofmap, set_vals, mode):
    """Assemble P1 mass matrix over a mesh into the PETSc matrix A"""
    connections, cell_ptr = mesh
    q = np.array([[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]], dtype=np.double)
    weights = np.full(3, 1.0 / 3.0, dtype=np.double)

    # Loop over cells
    N = np.empty(3, dtype=np.double)
    A_local = np.empty((3, 3), dtype=PETSc.ScalarType)
    for i, cell in enumerate(cell_ptr[:-1]):
        num_vertices = cell_ptr[i + 1] - cell_ptr[i]
        c = connections[cell:cell + num_vertices]
        cell_area = area(x[c[0]], x[c[1]], x[c[2]])

        # Loop over quadrature points
        A_local[:] = 0.0
        for j in range(q.shape[0]):
            N[0], N[1], N[2] = 1.0 - q[j, 0] - q[j, 1], q[j, 0], q[j, 1]
            for k in range(3):
                for l in range(3):
                    A_local[k, l] += weights[j] * cell_area * N[k] * N[l]

        rows = cols = dofmap[3 * i:3 * i + 3]
        set_vals(A, 3, rows.ctypes, 3, cols.ctypes, A_local.ctypes, mode)


def test_custom_mesh_loop_rank1():

    # Create mesh and function space
    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 64, 64)
    V = dolfin.FunctionSpace(mesh, ("Lagrange", 1))

    # Unpack mesh and dofmap data
    c = mesh.topology.connectivity(2, 0).connections()
    pos = mesh.topology.connectivity(2, 0).pos()
    geom = mesh.geometry.points
    dofs = V.dofmap().dof_array

    # Assemble with pure Numba function (two passes, first will include JIT overhead)
    b0 = dolfin.Function(V)
    for i in range(2):
        with b0.vector().localForm() as b:
            b.set(0.0)
            start = time.time()
            assemble_vector(np.asarray(b), (c, pos), geom, dofs)
            end = time.time()
            print("Time (numba, pass {}): {}".format(i, end - start))

    b0.vector().ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert(b0.vector().sum() == pytest.approx(1.0))

    # Test against generated code and general assembler
    v = dolfin.TestFunction(V)
    L = inner(1.0, v) * dx

    start = time.time()
    b1 = dolfin.fem.assemble_vector(L)
    end = time.time()
    print("Time (C++, pass 1):", end - start)

    with b1.localForm() as b_local:
        b_local.set(0.0)
    start = time.time()
    dolfin.fem.assemble_vector(b1, L)
    end = time.time()
    print("Time (C++, passs 2):", end - start)

    b1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert((b1 - b0.vector()).norm() == pytest.approx(0.0))

    # Assemble using generated tabulate_tensor kernel and Numba assembler
    b3 = dolfin.Function(V)
    ufc_form = dolfin.jit.ffc_jit(L)
    kernel = ufc_form.create_cell_integral(-1).tabulate_tensor
    for i in range(2):
        with b3.vector().localForm() as b:
            b.set(0.0)
            start = time.time()
            assemble_vector_ufc(np.asarray(b), kernel, (c, pos), geom, dofs)
            end = time.time()
            print("Time (numba/cffi, pass {}): {}".format(i, end - start))

    b3.vector().ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert((b3.vector() - b0.vector()).norm() == pytest.approx(0.0))


def test_custom_mesh_loop_ctypes_rank2():
    """Test numba assembler for bilinear form"""

    # Create mesh and function space
    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 64, 64)
    V = dolfin.FunctionSpace(mesh, ("Lagrange", 1))

    # Extract mesh and dofmap data
    c = mesh.topology.connectivity(2, 0).connections()
    pos = mesh.topology.connectivity(2, 0).pos()
    geom = mesh.geometry.points
    dofs = V.dofmap().dof_array

    # Generated case with general assembler
    u, v = dolfin.TrialFunction(V), dolfin.TestFunction(V)
    a = inner(u, v) * dx
    A0 = dolfin.fem.assemble_matrix(a)
    A0.assemble()
    A0.zeroEntries()

    start = time.time()
    dolfin.fem.assemble_matrix(A0, a)
    end = time.time()
    print("Time (C++, pass 2):", end - start)
    A0.assemble()

    # Custom case
    A1 = A0.copy()
    for i in range(2):
        A1.zeroEntries()
        mat = A1.handle
        start = time.time()
        assemble_matrix_ctypes(mat, (c, pos), geom, dofs, MatSetValues_ctypes, PETSc.InsertMode.ADD_VALUES)
        end = time.time()
        print("Time (numba, pass {}): {}".format(i, end - start))
        A1.assemble()

    assert (A0 - A1).norm() == pytest.approx(0.0, abs=1.0e-9)


@pytest.mark.parametrize("set_vals", [MatSetValues_abi, MatSetValues_api])
def test_custom_mesh_loop_cffi_rank2(set_vals):
    """Test numba assembler for bilinear form"""

    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 64, 64)
    V = dolfin.FunctionSpace(mesh, ("Lagrange", 1))

    # Test against generated code and general assembler
    u, v = dolfin.TrialFunction(V), dolfin.TestFunction(V)
    a = inner(u, v) * dx
    A0 = dolfin.fem.assemble_matrix(a)
    A0.assemble()

    A0.zeroEntries()
    start = time.time()
    dolfin.fem.assemble_matrix(A0, a)
    end = time.time()
    print("Time (C++, pass 2):", end - start)
    A0.assemble()

    # Unpack mesh and dofmap data
    c = mesh.topology.connectivity(2, 0).connections()
    pos = mesh.topology.connectivity(2, 0).pos()
    geom = mesh.geometry.points
    dofs = V.dofmap().dof_array

    A1 = A0.copy()
    for i in range(2):
        A1.zeroEntries()
        start = time.time()
        assemble_matrix_cffi(A1.handle, (c, pos), geom, dofs, set_vals, PETSc.InsertMode.ADD_VALUES)
        end = time.time()
        print("Time (Numba, pass {}): {}".format(i, end - start))
        A1.assemble()

    assert (A1 - A0).norm() == pytest.approx(0.0)
