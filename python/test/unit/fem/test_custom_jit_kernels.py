"""Unit tests for assembly with a numba kernel"""

# Copyright (C) 2018-2019 Chris N. Richardson and Michal Habera
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numba
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import (FunctionSpace, TimingType, UnitSquareMesh, cpp,
                     list_timings, Function)
from dolfinx_utils.test.skips import skip_if_complex
from dolfinx.fem import FormIntegrals

c_signature = numba.types.void(
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.types.double),
    numba.types.CPointer(numba.types.int32),
    numba.types.CPointer(numba.types.int32))


@numba.cfunc(c_signature, nopython=True)
def tabulate_tensor_A(A_, w_, c_, coords_, entity_local_index, cell_orientation):
    A = numba.carray(A_, (3, 3), dtype=PETSc.ScalarType)
    coordinate_dofs = numba.carray(coords_, (3, 2), dtype=np.float64)

    # Ke=∫Ωe BTe Be dΩ
    x0, y0 = coordinate_dofs[0, :]
    x1, y1 = coordinate_dofs[1, :]
    x2, y2 = coordinate_dofs[2, :]

    # 2x Element area Ae
    Ae = abs((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
    B = np.array(
        [y1 - y2, y2 - y0, y0 - y1, x2 - x1, x0 - x2, x1 - x0],
        dtype=PETSc.ScalarType).reshape(2, 3)
    A[:, :] = np.dot(B.T, B) / (2 * Ae)


@numba.cfunc(c_signature, nopython=True)
def tabulate_tensor_b(b_, w_, c_, coords_, local_index, orientation):
    b = numba.carray(b_, (3), dtype=PETSc.ScalarType)
    coordinate_dofs = numba.carray(coords_, (3, 2), dtype=np.float64)
    x0, y0 = coordinate_dofs[0, :]
    x1, y1 = coordinate_dofs[1, :]
    x2, y2 = coordinate_dofs[2, :]

    # 2x Element area Ae
    Ae = abs((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
    b[:] = Ae / 6.0


@numba.cfunc(c_signature, nopython=True)
def tabulate_tensor_b_coeff(b_, w_, c_, coords_, local_index, orientation):
    b = numba.carray(b_, (3), dtype=PETSc.ScalarType)
    w = numba.carray(w_, (1), dtype=PETSc.ScalarType)
    coordinate_dofs = numba.carray(coords_, (3, 2), dtype=np.float64)
    x0, y0 = coordinate_dofs[0, :]
    x1, y1 = coordinate_dofs[1, :]
    x2, y2 = coordinate_dofs[2, :]

    # 2x Element area Ae
    Ae = abs((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
    b[:] = w[0] * Ae / 6.0


def test_numba_assembly():
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 13, 13)
    V = FunctionSpace(mesh, ("Lagrange", 1))

    a = cpp.fem.Form([V._cpp_object, V._cpp_object])
    a.set_tabulate_tensor(FormIntegrals.Type.cell, -1, tabulate_tensor_A.address)
    a.set_tabulate_tensor(FormIntegrals.Type.cell, 12, tabulate_tensor_A.address)
    a.set_tabulate_tensor(FormIntegrals.Type.cell, 2, tabulate_tensor_A.address)

    L = cpp.fem.Form([V._cpp_object])
    L.set_tabulate_tensor(FormIntegrals.Type.cell, -1, tabulate_tensor_b.address)

    A = dolfinx.fem.assemble_matrix(a)
    A.assemble()
    b = dolfinx.fem.assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    Anorm = A.norm(PETSc.NormType.FROBENIUS)
    bnorm = b.norm(PETSc.NormType.N2)
    assert (np.isclose(Anorm, 56.124860801609124))
    assert (np.isclose(bnorm, 0.0739710713711999))

    list_timings(MPI.COMM_WORLD, [TimingType.wall])


def test_coefficient():
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 13, 13)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    DG0 = FunctionSpace(mesh, ("DG", 0))
    vals = Function(DG0)
    vals.vector.set(2.0)

    L = cpp.fem.Form([V._cpp_object])
    L.set_tabulate_tensor(FormIntegrals.Type.cell, -1, tabulate_tensor_b_coeff.address)
    L.set_coefficient(0, vals._cpp_object)

    b = dolfinx.fem.assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    bnorm = b.norm(PETSc.NormType.N2)
    print(bnorm)
    assert (np.isclose(bnorm, 2.0 * 0.0739710713711999))


@skip_if_complex
def test_cffi_assembly():
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 13, 13)
    V = FunctionSpace(mesh, ("Lagrange", 1))

    if mesh.mpi_comm().rank == 0:
        from cffi import FFI
        ffibuilder = FFI()
        ffibuilder.set_source("_cffi_kernelA", r"""
        #include <math.h>
        #include <stdalign.h>
        void tabulate_tensor_poissonA(double* restrict A, const double* w,
                                    const double* c,
                                    const double* restrict coordinate_dofs,
                                    const int* entity_local_index,
                                    const int* cell_orientation)
        {
        // Precomputed values of basis functions and precomputations
        // FE* dimensions: [entities][points][dofs]
        // PI* dimensions: [entities][dofs][dofs] or [entities][dofs]
        // PM* dimensions: [entities][dofs][dofs]
        alignas(32) static const double FE3_C0_D01_Q1[1][1][2] = { { { -1.0, 1.0 } } };
        // Unstructured piecewise computations
        const double J_c0 = coordinate_dofs[0] * FE3_C0_D01_Q1[0][0][0] + coordinate_dofs[2] * FE3_C0_D01_Q1[0][0][1];
        const double J_c3 = coordinate_dofs[1] * FE3_C0_D01_Q1[0][0][0] + coordinate_dofs[5] * FE3_C0_D01_Q1[0][0][1];
        const double J_c1 = coordinate_dofs[0] * FE3_C0_D01_Q1[0][0][0] + coordinate_dofs[4] * FE3_C0_D01_Q1[0][0][1];
        const double J_c2 = coordinate_dofs[1] * FE3_C0_D01_Q1[0][0][0] + coordinate_dofs[3] * FE3_C0_D01_Q1[0][0][1];
        alignas(32) double sp[20];
        sp[0] = J_c0 * J_c3;
        sp[1] = J_c1 * J_c2;
        sp[2] = sp[0] + -1 * sp[1];
        sp[3] = J_c0 / sp[2];
        sp[4] = -1 * J_c1 / sp[2];
        sp[5] = sp[3] * sp[3];
        sp[6] = sp[3] * sp[4];
        sp[7] = sp[4] * sp[4];
        sp[8] = J_c3 / sp[2];
        sp[9] = -1 * J_c2 / sp[2];
        sp[10] = sp[9] * sp[9];
        sp[11] = sp[8] * sp[9];
        sp[12] = sp[8] * sp[8];
        sp[13] = sp[5] + sp[10];
        sp[14] = sp[6] + sp[11];
        sp[15] = sp[12] + sp[7];
        sp[16] = fabs(sp[2]);
        sp[17] = sp[13] * sp[16];
        sp[18] = sp[14] * sp[16];
        sp[19] = sp[15] * sp[16];
        // UFLACS block mode: preintegrated
        A[0] = 0.5 * sp[19] + 0.5 * sp[18] + 0.5 * sp[18] + 0.5 * sp[17];
        A[1] = -0.5 * sp[19] + -0.5 * sp[18];
        A[2] = -0.5 * sp[18] + -0.5 * sp[17];
        A[3] = -0.5 * sp[19] + -0.5 * sp[18];
        A[4] = 0.5 * sp[19];
        A[5] = 0.5 * sp[18];
        A[6] = -0.5 * sp[18] + -0.5 * sp[17];
        A[7] = 0.5 * sp[18];
        A[8] = 0.5 * sp[17];
        }

        void tabulate_tensor_poissonL(double* restrict A, const double* w,
                                     const double* c,
                                     const double* restrict coordinate_dofs,
                                     const int* entity_local_index,
                                     const int* cell_orientation)
        {
        // Precomputed values of basis functions and precomputations
        // FE* dimensions: [entities][points][dofs]
        // PI* dimensions: [entities][dofs][dofs] or [entities][dofs]
        // PM* dimensions: [entities][dofs][dofs]
        alignas(32) static const double FE4_C0_D01_Q1[1][1][2] = { { { -1.0, 1.0 } } };
        // Unstructured piecewise computations
        const double J_c0 = coordinate_dofs[0] * FE4_C0_D01_Q1[0][0][0] + coordinate_dofs[2] * FE4_C0_D01_Q1[0][0][1];
        const double J_c3 = coordinate_dofs[1] * FE4_C0_D01_Q1[0][0][0] + coordinate_dofs[5] * FE4_C0_D01_Q1[0][0][1];
        const double J_c1 = coordinate_dofs[0] * FE4_C0_D01_Q1[0][0][0] + coordinate_dofs[4] * FE4_C0_D01_Q1[0][0][1];
        const double J_c2 = coordinate_dofs[1] * FE4_C0_D01_Q1[0][0][0] + coordinate_dofs[3] * FE4_C0_D01_Q1[0][0][1];
        alignas(32) double sp[4];
        sp[0] = J_c0 * J_c3;
        sp[1] = J_c1 * J_c2;
        sp[2] = sp[0] + -1 * sp[1];
        sp[3] = fabs(sp[2]);
        // UFLACS block mode: preintegrated
        A[0] = 0.1666666666666667 * sp[3];
        A[1] = 0.1666666666666667 * sp[3];
        A[2] = 0.1666666666666667 * sp[3];
        }
        """)
        ffibuilder.cdef("""
        void tabulate_tensor_poissonA(double* restrict A, const double* w,
                                    const double* c,
                                    const double* restrict coordinate_dofs,
                                    const int* entity_local_index,
                                    const int* cell_orientation);
        void tabulate_tensor_poissonL(double* restrict A, const double* w,
                                    const double* c,
                                    const double* restrict coordinate_dofs,
                                    const int* entity_local_index,
                                    const int* cell_orientation);
        """)

        ffibuilder.compile(verbose=True)

    mesh.mpi_comm().Barrier()
    from _cffi_kernelA import ffi, lib

    a = cpp.fem.Form([V._cpp_object, V._cpp_object])
    ptrA = ffi.cast("intptr_t", ffi.addressof(lib, "tabulate_tensor_poissonA"))
    a.set_tabulate_tensor(FormIntegrals.Type.cell, -1, ptrA)

    L = cpp.fem.Form([V._cpp_object])
    ptrL = ffi.cast("intptr_t", ffi.addressof(lib, "tabulate_tensor_poissonL"))
    L.set_tabulate_tensor(FormIntegrals.Type.cell, -1, ptrL)

    A = dolfinx.fem.assemble_matrix(a)
    A.assemble()
    b = dolfinx.fem.assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    Anorm = A.norm(PETSc.NormType.FROBENIUS)
    bnorm = b.norm(PETSc.NormType.N2)
    assert (np.isclose(Anorm, 56.124860801609124))
    assert (np.isclose(bnorm, 0.0739710713711999))

    list_timings(MPI.COMM_WORLD, [TimingType.wall])
