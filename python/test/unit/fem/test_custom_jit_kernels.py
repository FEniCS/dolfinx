"""Unit tests for assembly with Numba and CFFI kernels."""

# Copyright (C) 2018-2014 Chris N. Richardson, Michal Habera and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os
import sys

from mpi4py import MPI

import numpy as np
import pytest

import dolfinx
import dolfinx.utils
import ffcx.codegeneration.utils
from dolfinx import cpp as _cpp
from dolfinx import fem, la
from dolfinx.common import list_timings
from dolfinx.fem import Form, Function, IntegralType, form_cpp_class, functionspace
from dolfinx.mesh import create_unit_square

numba = pytest.importorskip("numba")
ufcx_signature = ffcx.codegeneration.utils.numba_ufcx_kernel_signature

sys.path.append(os.getcwd())


def tabulate_rank2(dtype, xdtype):
    @numba.cfunc(ufcx_signature(dtype, xdtype), nopython=True)
    def tabulate(A_, w_, c_, coords_, entity_local_index, cell_orientation):
        A = numba.carray(A_, (3, 3), dtype=dtype)
        coordinate_dofs = numba.carray(coords_, (3, 3), dtype=xdtype)

        # Ke=∫Ωe BTe Be dΩ
        x0, y0 = coordinate_dofs[0, :2]
        x1, y1 = coordinate_dofs[1, :2]
        x2, y2 = coordinate_dofs[2, :2]

        # 2x Element area Ae
        Ae = abs((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
        B = np.array([y1 - y2, y2 - y0, y0 - y1, x2 - x1, x0 - x2, x1 - x0], dtype=dtype).reshape(
            2, 3
        )
        A[:, :] = np.dot(B.T, B) / (2 * Ae)

    return tabulate


def tabulate_rank1(dtype, xdtype):
    @numba.cfunc(ufcx_signature(dtype, xdtype), nopython=True)
    def tabulate(b_, w_, c_, coords_, local_index, orientation):
        b = numba.carray(b_, (3), dtype=dtype)
        coordinate_dofs = numba.carray(coords_, (3, 3), dtype=xdtype)
        x0, y0 = coordinate_dofs[0, :2]
        x1, y1 = coordinate_dofs[1, :2]
        x2, y2 = coordinate_dofs[2, :2]

        # 2x Element area Ae
        Ae = abs((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
        b[:] = Ae / 6.0

    return tabulate


def tabulate_rank1_coeff(dtype, xdtype):
    @numba.cfunc(ufcx_signature(dtype, xdtype), nopython=True)
    def tabulate(b_, w_, c_, coords_, local_index, orientation):
        b = numba.carray(b_, (3), dtype=dtype)
        w = numba.carray(w_, (1), dtype=dtype)
        coordinate_dofs = numba.carray(coords_, (3, 3), dtype=xdtype)
        x0, y0 = coordinate_dofs[0, :2]
        x1, y1 = coordinate_dofs[1, :2]
        x2, y2 = coordinate_dofs[2, :2]

        # 2x Element area Ae
        Ae = abs((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
        b[:] = w[0] * Ae / 6.0

    return tabulate


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_numba_assembly(dtype):
    xdtype = np.real(dtype(0)).dtype
    k2 = tabulate_rank2(dtype, xdtype)
    k1 = tabulate_rank1(dtype, xdtype)
    mesh = create_unit_square(MPI.COMM_WORLD, 13, 13, dtype=xdtype)
    V = functionspace(mesh, ("Lagrange", 1))
    cells = np.arange(mesh.topology.index_map(mesh.topology.dim).size_local, dtype=np.int32)
    active_coeffs = np.array([], dtype=np.int8)
    integrals = {
        IntegralType.cell: [
            (-1, k2.address, cells, active_coeffs),
            (2, k2.address, np.arange(0), active_coeffs),
            (12, k2.address, np.arange(0), active_coeffs),
        ]
    }
    formtype = form_cpp_class(dtype)
    a = Form(
        formtype(
            [V._cpp_object, V._cpp_object], integrals, [], [], False, {}, mesh=mesh._cpp_object
        )
    )
    integrals = {IntegralType.cell: [(-1, k1.address, cells, active_coeffs)]}
    L = Form(formtype([V._cpp_object], integrals, [], [], False, {}, mesh=mesh._cpp_object))

    A = dolfinx.fem.assemble_matrix(a)
    A.scatter_reverse()
    b = dolfinx.fem.assemble_vector(L)
    b.scatter_reverse(dolfinx.la.InsertMode.add)

    Anorm = np.sqrt(A.squared_norm())
    bnorm = la.norm(b)
    assert np.isclose(Anorm, 56.124860801609124)
    assert np.isclose(bnorm, 0.0739710713711999)

    list_timings(MPI.COMM_WORLD)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_coefficient(dtype):
    xdtype = np.real(dtype(0)).dtype
    k1 = tabulate_rank1_coeff(dtype, xdtype)

    mesh = create_unit_square(MPI.COMM_WORLD, 13, 13, dtype=xdtype)
    V = functionspace(mesh, ("Lagrange", 1))
    DG0 = functionspace(mesh, ("DG", 0))
    vals = Function(DG0, dtype=dtype)
    vals.x.array[: vals.x.index_map.size_local] = 2

    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local + mesh.topology.index_map(tdim).num_ghosts
    active_coeffs = np.array([0], dtype=np.int8)
    integrals = {
        IntegralType.cell: [(1, k1.address, np.arange(num_cells, dtype=np.int32), active_coeffs)]
    }
    formtype = form_cpp_class(dtype)
    L = Form(
        formtype(
            [V._cpp_object], integrals, [vals._cpp_object], [], False, {}, mesh=mesh._cpp_object
        )
    )

    b = dolfinx.fem.assemble_vector(L)
    b.scatter_reverse(la.InsertMode.add)
    bnorm = la.norm(b)
    assert np.isclose(bnorm, 2.0 * 0.0739710713711999)


@pytest.mark.skip_in_parallel
def test_cffi_assembly():
    pytest.importorskip("cffi")
    mesh = create_unit_square(MPI.COMM_WORLD, 13, 13, dtype=np.float64)
    V = functionspace(mesh, ("Lagrange", 1))
    if mesh.comm.rank == 0:
        from cffi import FFI

        ffibuilder = FFI()
        ffibuilder.set_source(
            "_cffi_kernelA",
            r"""
        #include <math.h>
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
        static const double FE3_C0_D01_Q1[1][1][2] = { { { -1.0, 1.0 } } };
        // Unstructured piecewise computations
        const double J_c0 = coordinate_dofs[0] * FE3_C0_D01_Q1[0][0][0]
          + coordinate_dofs[3] * FE3_C0_D01_Q1[0][0][1];
        const double J_c3 = coordinate_dofs[1] * FE3_C0_D01_Q1[0][0][0]
          + coordinate_dofs[7] * FE3_C0_D01_Q1[0][0][1];
        const double J_c1 = coordinate_dofs[0] * FE3_C0_D01_Q1[0][0][0]
          + coordinate_dofs[6] * FE3_C0_D01_Q1[0][0][1];
        const double J_c2 = coordinate_dofs[1] * FE3_C0_D01_Q1[0][0][0]
          + coordinate_dofs[4] * FE3_C0_D01_Q1[0][0][1];
        double sp[20];
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
        static const double FE4_C0_D01_Q1[1][1][2] = { { { -1.0, 1.0 } } };
        // Unstructured piecewise computations
        const double J_c0 = coordinate_dofs[0] * FE4_C0_D01_Q1[0][0][0]
          + coordinate_dofs[3] * FE4_C0_D01_Q1[0][0][1];
        const double J_c3 = coordinate_dofs[1] * FE4_C0_D01_Q1[0][0][0]
          + coordinate_dofs[7] * FE4_C0_D01_Q1[0][0][1];
        const double J_c1 = coordinate_dofs[0] * FE4_C0_D01_Q1[0][0][0]
          + coordinate_dofs[6] * FE4_C0_D01_Q1[0][0][1];
        const double J_c2 = coordinate_dofs[1] * FE4_C0_D01_Q1[0][0][0]
          + coordinate_dofs[4] * FE4_C0_D01_Q1[0][0][1];
        double sp[4];
        sp[0] = J_c0 * J_c3;
        sp[1] = J_c1 * J_c2;
        sp[2] = sp[0] + -1 * sp[1];
        sp[3] = fabs(sp[2]);
        A[0] = 0.1666666666666667 * sp[3];
        A[1] = 0.1666666666666667 * sp[3];
        A[2] = 0.1666666666666667 * sp[3];
        }
        """,
        )
        ffibuilder.cdef(
            """
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
        """
        )

        ffibuilder.compile(verbose=True)

    mesh.comm.Barrier()
    from _cffi_kernelA import ffi, lib

    cells = np.arange(mesh.topology.index_map(mesh.topology.dim).size_local, dtype=np.int32)

    ptrA = ffi.cast("intptr_t", ffi.addressof(lib, "tabulate_tensor_poissonA"))
    active_coeffs = np.array([], dtype=np.int8)
    integrals = {IntegralType.cell: [(-1, ptrA, cells, active_coeffs)]}
    a = Form(
        _cpp.fem.Form_float64(
            [V._cpp_object, V._cpp_object], integrals, [], [], False, {}, mesh=mesh._cpp_object
        )
    )

    ptrL = ffi.cast("intptr_t", ffi.addressof(lib, "tabulate_tensor_poissonL"))
    integrals = {IntegralType.cell: [(-1, ptrL, cells, active_coeffs)]}
    L = Form(
        _cpp.fem.Form_float64([V._cpp_object], integrals, [], [], False, {}, mesh=mesh._cpp_object)
    )

    A = fem.assemble_matrix(a)
    A.scatter_reverse()
    assert np.isclose(np.sqrt(A.squared_norm()), 56.124860801609124)

    b = fem.assemble_vector(L)
    b.scatter_reverse(la.InsertMode.add)
    assert np.isclose(la.norm(b), 0.0739710713711999)
