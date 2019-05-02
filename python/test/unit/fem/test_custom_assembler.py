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
from dolfin_utils.test.skips import skip_if_complex
from ufl import dx, inner


@numba.jit(nopython=True, cache=True)
def area(x0, x1, x2) -> float:
    """Compute the area of a triangle embedded in 2D from the three vertices"""
    a = (x1[0] - x2[0])**2 + (x1[1] - x2[1])**2
    b = (x0[0] - x2[0])**2 + (x0[1] - x2[1])**2
    c = (x0[0] - x1[0])**2 + (x0[1] - x1[1])**2
    return math.sqrt(2 * (a * b + a * c + b * c) - (a**2 + b**2 + c**2)) / 4.0


def test_custom_mesh_loop_rank1():

    @numba.jit(nopython=True, cache=True)
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

    # CFFI - register complex types
    ffi = cffi.FFI()
    numba.cffi_support.register_type(ffi.typeof('double _Complex'),
                                     numba.types.complex128)
    numba.cffi_support.register_type(ffi.typeof('float _Complex'),
                                     numba.types.complex64)

    @numba.njit
    def assemble_vector_ufc(b, kernel, mesh, x, dofmap):
        """Assemble provided FFC/UFC kernel over a mesh into the array b"""
        connections, pos = mesh
        geometry = np.zeros((3, 2))
        coeffs = np.zeros(1, dtype=PETSc.ScalarType)
        b_local = np.zeros(3, dtype=PETSc.ScalarType)
        for i, cell in enumerate(pos[:-1]):
            num_vertices = pos[i + 1] - pos[i]
            c = connections[cell:cell + num_vertices]
            for j in range(3):
                for k in range(2):
                    geometry[j, k] = x[c[j], k]
            kernel(ffi.from_buffer(b_local), ffi.from_buffer(coeffs), ffi.from_buffer(geometry), 0)
            for j in range(3):
                b[dofmap[i * 3 + j]] += b_local[j]

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
    with b0.vector().localForm() as b:
        b.set(0.0)
        start = time.time()
        assemble_vector(np.asarray(b), (c, pos), geom, dofs)
        end = time.time()
        print("Time (numba, pass 1):", end - start)

    with b0.vector().localForm() as b:
        b.set(0.0)
        start = time.time()
        assemble_vector(np.asarray(b), (c, pos), geom, dofs)
        end = time.time()
        print("Time (numba, pass 2):", end - start)

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
    with b3.vector().localForm() as b:
        b.set(0.0)
        start = time.time()
        assemble_vector_ufc(np.asarray(b), kernel, (c, pos), geom, dofs)
        end = time.time()
        print("Time (numba/cffi, pass 1):", end - start)

    with b3.vector().localForm() as b:
        b.set(0.0)
        start = time.time()
        assemble_vector_ufc(np.asarray(b), kernel, (c, pos), geom, dofs)
        end = time.time()
        print("Time (numba/cffi, pass 2):", end - start)

    b3.vector().ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert((b3.vector() - b0.vector()).norm() == pytest.approx(0.0))


def test_custom_mesh_loop_ctypes_rank2():
    """Test numba assembler for bilinear form"""

    # Load PETSc library
    ctypes.util.find_library("petsc")
    petsc_lib_name = ctypes.util.find_library("petsc")
    if petsc_lib_name is not None:
        petsc_lib = ctypes.CDLL(petsc_lib_name)
    else:
        petsc_dir = os.environ.get('PETSC_DIR', None)
        try:
            petsc_lib = ctypes.CDLL(os.path.join(petsc_dir, "lib", "libpetsc.so"))
        except OSError:
            petsc_lib = ctypes.CDLL(os.path.join(petsc_dir, "lib", "libpetsc.dylib"))
        except OSError:
            print("Could not load PETSc library for CFFI (ABI mode).")
            raise

    # Get the PETSc MatSetValuesLocal function
    MatSetValues = petsc_lib.MatSetValuesLocal
    MatSetValues.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(
        ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int)
    ADD_VALUES = PETSc.InsertMode.ADD_VALUES
    del petsc_lib

    @numba.njit
    def assemble_matrix(A, mesh, x, dofmap, set_vals):
        """Assemble P1 mass matrix over a mesh into the PETSc matrix A"""

        def shape_functions(q):
            """Compute shape functions at a point"""
            return 1.0 - q[0] - q[1], q[0], q[1]

        # Mesh data
        connections, pos = mesh

        # Quadrature points and weights
        q_points = np.array([[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]], dtype=np.double)
        weights = np.full(3, 1.0 / 3.0, dtype=np.double)

        # Loop over cells
        _A = np.empty((3, 3), dtype=PETSc.ScalarType)
        for i, cell in enumerate(pos[:-1]):
            num_vertices = pos[i + 1] - pos[i]
            c = connections[cell:cell + num_vertices]
            cell_area = area(x[c[0]], x[c[1]], x[c[2]])

            # Loop over quadrature points
            _A[:] = 0.0
            for j in range(q_points.shape[0]):

                N0, N1, N2 = shape_functions(q_points[j])

                _A[0, 0] += weights[j] * cell_area * N0 * N0
                _A[0, 1] += weights[j] * cell_area * N0 * N1
                _A[0, 2] += weights[j] * cell_area * N0 * N2

                _A[1, 0] += weights[j] * cell_area * N1 * N0
                _A[1, 1] += weights[j] * cell_area * N1 * N1
                _A[1, 2] += weights[j] * cell_area * N1 * N2

                _A[2, 0] += weights[j] * cell_area * N2 * N0
                _A[2, 1] += weights[j] * cell_area * N2 * N1
                _A[2, 2] += weights[j] * cell_area * N2 * N2

            # Add to global tensor
            rows = cols = dofmap[3 * i:3 * i + 3].ctypes
            set_vals(A, 3, rows, 3, cols, _A.ctypes, ADD_VALUES)

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
    A1.zeroEntries()
    mat = A1.handle
    assemble_matrix(mat, (c, pos), geom, dofs, MatSetValues)
    A1.assemble()

    A1.zeroEntries()
    start = time.time()
    assemble_matrix(mat, (c, pos), geom, dofs, MatSetValues)
    end = time.time()
    print("Time (numba, pass 2):", end - start)
    A1.assemble()

    assert (A0 - A1).norm() == pytest.approx(0.0, abs=1.0e-9)


@skip_if_complex
def test_custom_mesh_loop_cffi_rank2():
    """Test numba assembler for bilinear form

    Some work is required to get this working with complex types, and possibly
    64-bit indices.

    """

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

    # Make MatSetValuesLocal from PETSc available via cffi
    if mesh.mpi_comm().Get_rank() == 0:
        os.environ["CC"] = "mpicc"
        petsc_dir = os.environ.get('PETSC_DIR', None)
        ffibuilder = cffi.FFI()
        ffibuilder.cdef("""
            typedef int... PetscInt;
            typedef int... PetscErrorCode;
            typedef ... PetscScalar;
            typedef int... InsertMode;
            PetscErrorCode MatSetValuesLocal(void* mat, PetscInt nrow, const PetscInt* irow,
                                PetscInt ncol, const PetscInt* icol,
                                const PetscScalar* y, InsertMode addv);
        """)
        ffibuilder.set_source("_petsc_cffi", """
            # include "petscmat.h"
        """,
                              libraries=['petsc'],
                              include_dirs=[os.path.join(petsc_dir, 'include')],
                              library_dirs=[os.path.join(petsc_dir, 'lib')],
                              extra_compile_args=[])
        ffibuilder.compile(verbose=False)

    mesh.mpi_comm().barrier()

    spec = importlib.util.find_spec('_petsc_cffi')
    if spec is None:
        raise ImportError("Failed to find CFFI generated module")
    module = importlib.util.module_from_spec(spec)

    numba.cffi_support.register_module(module)
    add_values = module.lib.MatSetValuesLocal
    numba.cffi_support.register_type(module.ffi.typeof("PetscScalar"), numba.types.float64)

    # See https://github.com/numba/numba/issues/4036 for why we need 'sink'
    @numba.njit
    def sink(*args):
        pass

    @numba.njit
    def assemble_matrix(A, mesh, x, dofmap, set_vals, mode):
        """Assemble P1 mass matrix over a mesh into the PETSc matrix A"""

        def shape_functions(q):
            """Compute shape functions at a point"""
            return 1.0 - q[0] - q[1], q[0], q[1]

        # Mesh data
        connections, pos = mesh

        # Quadrature points and weights
        q_points = np.array([[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]], dtype=np.double)
        weights = np.full(3, 1.0 / 3.0, dtype=np.double)

        # Loop over cells
        _A = np.empty((3, 3), dtype=PETSc.ScalarType)
        for i, cell in enumerate(pos[:-1]):
            num_vertices = pos[i + 1] - pos[i]
            c = connections[cell:cell + num_vertices]
            cell_area = area(x[c[0]], x[c[1]], x[c[2]])

            # Loop over quadrature points
            _A[:] = 0.0
            for j in range(q_points.shape[0]):

                N0, N1, N2 = shape_functions(q_points[j])

                _A[0, 0] += weights[j] * cell_area * N0 * N0
                _A[0, 1] += weights[j] * cell_area * N0 * N1
                _A[0, 2] += weights[j] * cell_area * N0 * N2

                _A[1, 0] += weights[j] * cell_area * N1 * N0
                _A[1, 1] += weights[j] * cell_area * N1 * N1
                _A[1, 2] += weights[j] * cell_area * N1 * N2

                _A[2, 0] += weights[j] * cell_area * N2 * N0
                _A[2, 1] += weights[j] * cell_area * N2 * N1
                _A[2, 2] += weights[j] * cell_area * N2 * N2

            # Add to global tensor
            rows = cols = module.ffi.from_buffer(dofmap[3 * i:3 * i + 3])
            set_vals(A, 3, rows, 3, cols, module.ffi.from_buffer(_A), mode)
        sink(_A, dofmap)

    # Unpack mesh and dofmap data
    c = mesh.topology.connectivity(2, 0).connections()
    pos = mesh.topology.connectivity(2, 0).pos()
    geom = mesh.geometry.points
    dofs = V.dofmap().dof_array

    # First assembly
    A1 = A0.copy()
    A1.zeroEntries()
    assemble_matrix(A1.handle, (c, pos), geom, dofs, add_values, PETSc.InsertMode.ADD_VALUES)
    A1.assemble()

    # Second assembly
    A1.zeroEntries()
    start = time.time()
    assemble_matrix(A1.handle, (c, pos), geom, dofs, add_values, PETSc.InsertMode.ADD_VALUES)
    end = time.time()
    print("Time (Numba, pass 2):", end - start)
    A1.assemble()

    assert (A1 - A0).norm() == pytest.approx(0.0)


@skip_if_complex
def test_custom_mesh_loop_cffi_abi_rank2():
    """Test numba assembler for bilinear form

    Some work is required to get this working with complex types, and possibly
    64-bit indices.

    """

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

    # Make MatSetValuesLocal from PETSc available via cffi
    ffi = cffi.FFI()
    ffi.cdef("""
        int MatSetValuesLocal(void* mat, int nrow, const int* irow, int ncol, const int* icol,
                              const double* y, int addv);
    """)

    petsc_lib = ctypes.util.find_library("petsc")
    if petsc_lib is None:
        petsc_dir = os.environ.get('PETSC_DIR', None)
        try:
            petsc_lib = ffi.dlopen(os.path.join(petsc_dir, "lib", "libpetsc.so"))
        except OSError:
            petsc_lib = ffi.dlopen(os.path.join(petsc_dir, "lib", "libpetsc.dylib"))
        except OSError:
            print("Could not load PETSc library for CFFI (ABI mode).")
            raise
    add_values = petsc_lib.MatSetValuesLocal

    # See https://github.com/numba/numba/issues/4036 for why we need 'sink'
    @numba.njit
    def sink(*args):
        pass

    @numba.njit
    def assemble_matrix(A, mesh, x, dofmap, set_vals, mode):
        """Assemble P1 mass matrix over a mesh into the PETSc matrix A"""

        def shape_functions(q):
            """Compute shape functions at a point"""
            return 1.0 - q[0] - q[1], q[0], q[1]

        # Mesh data
        connections, pos = mesh

        # Quadrature points and weights
        q_points = np.array([[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]], dtype=np.double)
        weights = np.full(3, 1.0 / 3.0, dtype=np.double)

        # Loop over cells
        _A = np.empty((3, 3), dtype=PETSc.ScalarType)
        for i, cell in enumerate(pos[:-1]):
            num_vertices = pos[i + 1] - pos[i]
            c = connections[cell:cell + num_vertices]
            cell_area = area(x[c[0]], x[c[1]], x[c[2]])

            # Loop over quadrature points
            _A[:] = 0.0
            for j in range(q_points.shape[0]):

                N0, N1, N2 = shape_functions(q_points[j])

                _A[0, 0] += weights[j] * cell_area * N0 * N0
                _A[0, 1] += weights[j] * cell_area * N0 * N1
                _A[0, 2] += weights[j] * cell_area * N0 * N2

                _A[1, 0] += weights[j] * cell_area * N1 * N0
                _A[1, 1] += weights[j] * cell_area * N1 * N1
                _A[1, 2] += weights[j] * cell_area * N1 * N2

                _A[2, 0] += weights[j] * cell_area * N2 * N0
                _A[2, 1] += weights[j] * cell_area * N2 * N1
                _A[2, 2] += weights[j] * cell_area * N2 * N2

            # Add to global tensor
            rows = cols = ffi.from_buffer(dofmap[3 * i:3 * i + 3])
            set_vals(A, 3, rows, 3, cols, ffi.from_buffer(_A), mode)
        sink(_A, dofmap)

    # Unpack mesh and dofmap data
    c = mesh.topology.connectivity(2, 0).connections()
    pos = mesh.topology.connectivity(2, 0).pos()
    geom = mesh.geometry.points
    dofs = V.dofmap().dof_array

    # First assembly
    A1 = A0.copy()
    A1.zeroEntries()
    assemble_matrix(A1.handle, (c, pos), geom, dofs, add_values, PETSc.InsertMode.ADD_VALUES)
    A1.assemble()

    # Second assembly
    A1.zeroEntries()
    start = time.time()
    assemble_matrix(A1.handle, (c, pos), geom, dofs, add_values, PETSc.InsertMode.ADD_VALUES)
    end = time.time()
    print("Time (Numba, pass 2):", end - start)
    A1.assemble()

    assert (A1 - A0).norm() == pytest.approx(0.0)
