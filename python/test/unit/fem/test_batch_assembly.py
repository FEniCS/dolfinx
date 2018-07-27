# Copyright (C) 2018 Fabian Löschner
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests assembly with cell batches"""

import numpy

import dolfin
from dolfin import DOLFIN_EPS, MPI

import ufl
from ufl import inner, grad, dx


def test_batch_assembly():
    def assemble_test(cell_batch_size: int):
        mesh = dolfin.UnitCubeMesh(MPI.comm_world, 2, 3, 4)
        element = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)

        Q = dolfin.function.functionspace.FunctionSpace(mesh, element)

        u = dolfin.function.argument.TrialFunction(Q)
        v = dolfin.function.argument.TestFunction(Q)

        def boundary(x):
            return numpy.sum(numpy.logical_or(x < DOLFIN_EPS, x > 1.0 - DOLFIN_EPS), axis=1) > 0

        u_bc = dolfin.function.constant.Constant(50.0)
        bc = dolfin.fem.dirichletbc.DirichletBC(Q, u_bc, boundary)

        f = dolfin.function.constant.Constant(1.0)

        a = inner(grad(u), grad(v)) * dx
        L = f * v * dx

        # Create assembler
        assembler = dolfin.fem.assembling.Assembler([[a]], [L], [bc])

        A, b = assembler.assemble(mat_type=dolfin.cpp.fem.Assembler.BlockType.monolithic)

        return A, b

    A1, b1 = assemble_test(cell_batch_size=1)
    A4, b4 = assemble_test(cell_batch_size=4)

    A1norm = A1.norm(dolfin.cpp.la.Norm.frobenius)
    b1norm = b1.norm(dolfin.cpp.la.Norm.l2)

    A4norm = A4.norm(dolfin.cpp.la.Norm.frobenius)
    b4norm = b4.norm(dolfin.cpp.la.Norm.l2)

    assert(numpy.isclose(A1norm, A4norm))
    assert(numpy.isclose(b1norm, b4norm))

    print(A1norm)
