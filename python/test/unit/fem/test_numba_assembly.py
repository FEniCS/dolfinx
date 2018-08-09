"""Unit tests for assembly with a numba kernel"""

# Copyright (C) 2018 Chris N. Richardson
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import (cpp, UnitSquareMesh, MPI, FunctionSpace,
                    dx, dot, grad, TestFunction, TrialFunction, list_timings,
                    TimingType)
from numba import cfunc, types, carray, typeof
from dolfin.jit.jit import ffc_jit
import numpy as np
from petsc4py.PETSc import ScalarType


def tabulate_tensor_A(A_, w_, coords_, cell_orientation):
    A = carray(A_, (3, 3), dtype=ScalarType)
    coordinate_dofs = carray(coords_, (3, 2), dtype=np.float64)

    # Ke=∫Ωe BTe Be dΩ
    x0, y0 = coordinate_dofs[0, :]
    x1, y1 = coordinate_dofs[1, :]
    x2, y2 = coordinate_dofs[2, :]

    # 2x Element area Ae
    Ae = abs((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))

    B = np.array(
        [y1 - y2, y2 - y0, y0 - y1, x2 - x1, x0 - x2, x1 - x0],
        dtype=ScalarType).reshape(2, 3)

    A[:, :] = np.dot(B.T, B) / (2 * Ae)


def tabulate_tensor_b(b_, w_, coords_, cell_orientation):
    b = carray(b_, (3), dtype=ScalarType)
    coordinate_dofs = carray(coords_, (3, 2), dtype=np.float64)
    x0, y0 = coordinate_dofs[0, :]
    x1, y1 = coordinate_dofs[1, :]
    x2, y2 = coordinate_dofs[2, :]

    # 2x Element area Ae
    Ae = abs((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
    b[:] = Ae / 6.0


def test_numba_assembly():
    mesh = UnitSquareMesh(MPI.comm_world, 13, 13)
    Q = FunctionSpace(mesh, "Lagrange", 1)

    u = TrialFunction(Q)
    v = TestFunction(Q)

    a = cpp.fem.Form([Q._cpp_object, Q._cpp_object])
    L = cpp.fem.Form([Q._cpp_object])

    sig = types.void(
        types.CPointer(typeof(ScalarType())),
        types.CPointer(types.CPointer(typeof(ScalarType()))),
        types.CPointer(types.double), types.intc)

    fnA = cfunc(sig, cache=True)(tabulate_tensor_A)
    a.set_cell_tabulate(0, fnA.address)

    fnb = cfunc(sig, cache=True)(tabulate_tensor_b)
    L.set_cell_tabulate(0, fnb.address)

    if (False):
        ufc_form = ffc_jit(dot(grad(u), grad(v)) * dx)
        ufc_form = cpp.fem.make_ufc_form(ufc_form[0])
        a = cpp.fem.Form(ufc_form, [Q._cpp_object, Q._cpp_object])
        ufc_form = ffc_jit(v * dx)
        ufc_form = cpp.fem.make_ufc_form(ufc_form[0])
        L = cpp.fem.Form(ufc_form, [Q._cpp_object])

    assembler = cpp.fem.Assembler([[a]], [L], [])
    A = assembler.assemble_matrix(cpp.fem.Assembler.BlockType.monolithic)
    b = assembler.assemble_vector(cpp.fem.Assembler.BlockType.monolithic)

    Anorm = A.norm(cpp.la.Norm.frobenius)
    bnorm = b.norm(cpp.la.Norm.l2)

    print(Anorm, bnorm)

    assert (np.isclose(Anorm, 56.124860801609124))
    assert (np.isclose(bnorm, 0.0739710713711999))

    list_timings([TimingType.wall])
