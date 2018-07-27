# Copyright (C) 2018 Fabian Löschner
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests assembly with cell batches"""

import numpy

import dolfin
from dolfin import DOLFIN_EPS, MPI
from dolfin.jit.jit import ffc_jit
from dolfin.la import PETScMatrix, PETScVector

import ufl
from ufl import dot, grad, dx


def test_batch_assembly():
    def assemble_test(cell_batch_size: int):
        mesh = dolfin.UnitCubeMesh(MPI.comm_world, 2, 3, 4)
        Q = dolfin.FunctionSpace(mesh, "Lagrange", 1)

        u = ufl.TrialFunction(Q)
        v = ufl.TestFunction(Q)

        # Define the boundary: vertices where any component is in machine precision accuracy 0 or 1
        def boundary(x):
            return numpy.sum(numpy.logical_or(x < DOLFIN_EPS, x > 1.0 - DOLFIN_EPS), axis=1) > 0

        u0 = dolfin.Constant(0.0)
        bc = dolfin.DirichletBC(Q, u0, boundary)

        # Initialize bilinear form and rhs
        a = dolfin.cpp.fem.Form([Q._cpp_object, Q._cpp_object])
        L = dolfin.cpp.fem.Form([Q._cpp_object])

        print("Form compilation...")

        # Bilinear form
        jit_result = ffc_jit(dot(grad(u), grad(v)) * dx,
                             form_compiler_parameters={"cell_batch_size": cell_batch_size,
                                                       "enable_cross_element_gcc_ext": True})
        ufc_form = dolfin.cpp.fem.make_ufc_form(jit_result[0])
        a = dolfin.cpp.fem.Form(ufc_form, [Q._cpp_object, Q._cpp_object])

        # Rhs
        f = dolfin.Expression("2.0", element=Q.ufl_element())
        jit_result = ffc_jit(f * v * dx,
                             form_compiler_parameters={"cell_batch_size": cell_batch_size,
                                                       "enable_cross_element_gcc_ext": True})
        ufc_form = dolfin.cpp.fem.make_ufc_form(jit_result[0])
        L = dolfin.cpp.fem.Form(ufc_form, [Q._cpp_object])
        # Attach rhs expression as coefficient
        L.set_coefficient(0, f._cpp_object)

        assembler = dolfin.cpp.fem.Assembler([[a]], [L], [bc])

        A = PETScMatrix()
        b = PETScVector()

        print("Running assembly...")
        assembler.assemble(A, dolfin.cpp.fem.Assembler.BlockType.monolithic)
        assembler.assemble(b, dolfin.cpp.fem.Assembler.BlockType.monolithic)

        return A, b

    A1, b1 = assemble_test(cell_batch_size=1)
    A4, b4 = assemble_test(cell_batch_size=4)

    A1norm = A1.norm(dolfin.cpp.la.Norm.frobenius)
    b1norm = b1.norm(dolfin.cpp.la.Norm.l2)

    A4norm = A4.norm(dolfin.cpp.la.Norm.frobenius)
    b4norm = b4.norm(dolfin.cpp.la.Norm.l2)

    assert(numpy.isclose(A1norm, A4norm))
    assert(numpy.isclose(b1norm, b4norm))
