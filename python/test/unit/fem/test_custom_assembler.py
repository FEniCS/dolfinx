# Copyright (C) 2019 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tests for custom Python assemblers"""

import math
import time

import cffi
import numba
import numba.cffi_support
import numpy as np
import pytest
from numba import jit
from petsc4py import PETSc

import dolfin
from ufl import dx, inner


@jit(nopython=True, cache=True)
def area(x0, x1, x2):
    a = (x1[0] - x2[0])**2 + (x1[1] - x2[1])**2
    b = (x0[0] - x2[0])**2 + (x0[1] - x2[1])**2
    c = (x0[0] - x1[0])**2 + (x0[1] - x1[1])**2
    return math.sqrt(2 * (a * b + a * c + b * c) - (a**2 + b**2 + c**2)) / 4.0


def test_custom_mesh_loop_rank1():

    @jit(nopython=True, cache=True)
    def assemble_vector(b, mesh, x, dofmap):
        """Assemble over a mesh into the array b"""
        connections, pos = mesh
        q0, q1 = 1 / 3.0, 1 / 3.0
        for i, cell in enumerate(pos[:-1]):
            num_vertices = pos[i + 1] - pos[i]
            c = connections[cell:cell + num_vertices]
            A = area(x[c[0]], x[c[1]], x[c[2]])
            b[dofmap[i * 3 + 0]] += A * (1.0 - q0 - q1)
            b[dofmap[i * 3 + 1]] += A * q0
            b[dofmap[i * 3 + 2]] += A * q1

    ffi = cffi.FFI()
    numba.cffi_support.register_type(ffi.typeof('double _Complex'),
                                     numba.types.complex128)
    numba.cffi_support.register_type(ffi.typeof('float _Complex'),
                                     numba.types.complex64)

    @jit(nopython=True)
    def assemble_vector_ufc(b, kernel, mesh, x, dofmap):
        """Assemble provided kernel over a mesh into the array b"""
        connections, pos = mesh
        b_local = np.zeros(3, dtype=PETSc.ScalarType)
        geometry = np.zeros((3, 2))
        coeffs = np.zeros(1, dtype=PETSc.ScalarType)
        for i, cell in enumerate(pos[:-1]):
            num_vertices = pos[i + 1] - pos[i]
            c = connections[cell:cell + num_vertices]
            for j in range(3):
                for k in range(2):
                    geometry[j, k] = x[c[j], k]
            kernel(ffi.from_buffer(b_local), ffi.from_buffer(coeffs), ffi.from_buffer(geometry), 0)
            for j in range(3):
                b[dofmap[i * 3 + j]] += b_local[j]

    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 64, 64)
    V = dolfin.FunctionSpace(mesh, ("Lagrange", 1))
    b0 = dolfin.Function(V)

    c = mesh.topology.connectivity(2, 0).connections()
    pos = mesh.topology.connectivity(2, 0).pos()
    geom = mesh.geometry.points
    dofs = V.dofmap().dof_array

    with b0.vector().localForm() as b:
        b.set(0.0)
        _b = np.asarray(b)
        start = time.time()
        assemble_vector(_b, (c, pos), geom, dofs)
        end = time.time()
        print("Time (numba, 1):", end - start)

    with b0.vector().localForm() as b:
        b.set(0.0)
        _b = np.asarray(b)
        start = time.time()
        assemble_vector(_b, (c, pos), geom, dofs)
        end = time.time()
        print("Time (numba, 2):", end - start)

    b0.vector().ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert(b0.vector().sum() == pytest.approx(1.0))

    # Test against generated code and general assembler
    v = dolfin.TestFunction(V)
    L = inner(1.0, v) * dx

    start = time.time()
    b1 = dolfin.fem.assemble_vector(L)
    end = time.time()
    print("Time (C++, 1):", end - start)

    with b1.localForm() as b_local:
        b_local.set(0.0)
    start = time.time()
    dolfin.fem.assemble_vector(b1, L)
    end = time.time()
    print("Time (C++, 2):", end - start)

    b1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert(b1.sum() == pytest.approx(1.0))

    b2 = b1 - b0.vector()
    assert(b2.norm() == pytest.approx(0.0))

    # Complex not supported yet
    # cffi_support.register_type('double _Complex', numba.types.complex128)
    # if dolfin.has_petsc_complex:
    #     return

    # Assemble using generated tabulate_tensor kernel
    b3 = dolfin.Function(V)
    ufc_form = dolfin.jit.ffc_jit(L)
    kernel = ufc_form.create_cell_integral(-1).tabulate_tensor
    with b3.vector().localForm() as b:
        b.set(0.0)
        _b = np.asarray(b)
        start = time.time()
        assemble_vector_ufc(_b, kernel, (c, pos), geom, dofs)
        end = time.time()
        print("Time (numba/cffi, 1):", end - start)

    with b3.vector().localForm() as b:
        b.set(0.0)
        _b = np.asarray(b)
        start = time.time()
        assemble_vector_ufc(_b, kernel, (c, pos), geom, dofs)
        end = time.time()
        print("Time (numba/cffi, 2):", end - start)

    b3.vector().ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert(b3.vector().sum() == pytest.approx(1.0))


def test_custom_mesh_loop_rank2():

    if dolfin.has_petsc_complex:
        return

    import os
    import ctypes
    import ctypes.util
    petsc_dir = os.environ.get('PETSC_DIR', None)
    try:
        petsc_lib = ctypes.CDLL(petsc_dir + "/lib/libpetsc.so")
    except OSError:
        petsc_lib = ctypes.CDLL(petsc_dir + "/lib/libpetsc.dylib")
    MatSetValues = petsc_lib.MatSetValuesLocal
    MatSetValues.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(
        ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.c_int)
    ADD_VALUES = PETSc.InsertMode.ADD_VALUES
    del petsc_lib

    @jit(nopython=True)
    def assemble_matrix(A, mesh, x, dofmap, set_vals):
        """Assemble over a mesh into the PETSc matrix A"""
        def shape_functions(q):
            return 1.0 - q[0] - q[1], q[0], q[1]

        # Mesh data
        connections, pos = mesh

        # Quadrature points and weights
        q_points = np.empty((3, 2), dtype=np.double)
        q_points[0, 0], q_points[0, 1] = 0.5, 0.0
        q_points[1, 0], q_points[1, 1] = 0.5, 0.5
        q_points[2, 0], q_points[2, 1] = 0.0, 0.5
        weights = np.full(3, 1.0 / 3.0, dtype=np.double)

        # Loop over cells
        _A = np.empty((3, 3), dtype=np.double)
        for i, cell in enumerate(pos[:-1]):
            num_vertices = pos[i + 1] - pos[i]
            c = connections[cell:cell + num_vertices]
            cell_area = area(x[c[0]], x[c[1]], x[c[2]])
            _A[:] = 0.0

            # Loop over quadrature points
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

    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 64, 64)
    V = dolfin.FunctionSpace(mesh, ("Lagrange", 1))
    c = mesh.topology.connectivity(2, 0).connections()
    pos = mesh.topology.connectivity(2, 0).pos()
    geom = mesh.geometry.points
    dofs = V.dofmap().dof_array

    # Test against generated code and general assembler
    u, v = dolfin.TrialFunction(V), dolfin.TestFunction(V)
    a = inner(u, v) * dx
    A0 = dolfin.fem.assemble_matrix(a)
    A0.assemble()
    A0.zeroEntries()

    start = time.time()
    dolfin.fem.assemble_matrix(A0, a)
    end = time.time()
    print("Time (C++, 2):", end - start)
    A0.assemble()

    A1 = A0.copy()
    A1.zeroEntries()
    mat = A1.handle
    assemble_matrix(mat, (c, pos), geom, dofs, MatSetValues)
    A1.assemble()

    A1.zeroEntries()
    start = time.time()
    assemble_matrix(mat, (c, pos), geom, dofs, MatSetValues)
    end = time.time()
    print("Time (numba, 2):", end - start)
    A1.assemble()

    assert (A0 - A1).norm() == pytest.approx(0.0, abs=1.0e-9)

    # assemble_matrix(mat, PETSc.InsertMode.ADD)
    # A.assemble()
    # A.view()
    # print("Test norm: ", A.norm())

    # libpetsc_path = ctypes.util.find_library('petsc')
    # print(libpetsc_path)

#     from cffi import FFI
#     ffibuilder = FFI()

#     ffibuilder.cdef("""
#         typedef struct{...;} Mat;

#         /*typedef ... Mat;*/
#         /* typedef ... InsertMode; */
#         int MatSetValuesLocal(Mat A, int nrow, const int* irow, int ncol, const int* icol,
#                               const double* y, ...);
#         int mysum(int x);
#         void mymat(Mat mat, uintptr_t x);
#     """)

#     ffibuilder.set_source("_petsc_cffi",
#                           """
#      # include "petscmat.h"   // the C header of the library
#      int mysum(int x) { return 2 + x ; }
#      void mymat(Mat mat, uintptr_t x) { mat = NULL; }
# """,
#                           libraries=['petsc'],
#                           include_dirs=['/Users/garth/local/packages/petsc-dev/include'],
#                           library_dirs=["/Users/garth/local/packages/petsc-dev/lib"])

#     ffibuilder.compile(verbose=True)


#     from _petsc_cffi import ffi, lib

#     test = ffi.typeof("Mat")
#     test = ffi.sizeof("Mat")
#     print(test, type(A.handle))
#     p = ffi.new("Mat *")
#     print(p)

#     # test1 = ffi.cast("Mat", A.handle)
#     # test = lib.mysum(2)
#     # print(test)

#     # test0 = ffi.typeof("Mat")
#     # testb = ffi.new_handle(A)
#     # print(testb)
#     # # test1 = ffi.new(test0)
#     # print("Boo:", test0)
#     # # print("Boo:", ffi.sizeof("Mat"))

#     # # Ah = A.handle
#     # # print(type(A.handle))

#     # # test = ffi.cast("Mat", A.handle)
#     # p_handle = ffi.new("Mat *")
#     # print(type(p_handle))

#     # junk = ffi.cast("uintptr_t", A.handle)
#     # print("junk", junk)

#     # pos = np.zeros(1, dtype=np.uint8)
#     # lib.MatSetValuesLocal(junk, 1, ffi.from_buffer(pos), 1, ffi.from_buffer(pos),
#     #                       ffi.from_buffer(vals), PETSc.InsertMode.ADD)
#     # test=PETSc.InsertMode.ADD
#     # print(test, type(test))


#     # assemble_matrix(lib.MatSetValuesLocal)
