# Copyright (C) 2018 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly"""

import math
import os

import numpy
import pytest

import dolfin
import ufl
from dolfin_utils.test import skip_in_parallel
from ufl import dx


def test_matrix_assembly_block():
    """Test assembly of block matrices and vectors into (a) monolithic
    blocked structures, PETSc Nest structures, and monolithic structures.
    """

    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 2, 5)

    p0, p1 = 1, 2
    P0 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p0)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p1)

    V0 = dolfin.function.functionspace.FunctionSpace(mesh, P0)
    V1 = dolfin.function.functionspace.FunctionSpace(mesh, P1)

    def boundary(x):
        return numpy.logical_or(x[:, 0] < 1.0e-6, x[:, 0] > 1.0 - 1.0e-6)

    u_bc = dolfin.function.constant.Constant(50.0)
    bc = dolfin.fem.dirichletbc.DirichletBC(V1, u_bc, boundary)

    # Define variational problem
    u, p = dolfin.function.argument.TrialFunction(
        V0), dolfin.function.argument.TrialFunction(V1)
    v, q = dolfin.function.argument.TestFunction(
        V0), dolfin.function.argument.TestFunction(V1)
    f = dolfin.function.constant.Constant(1.0)
    g = dolfin.function.constant.Constant(-3.0)
    zero = dolfin.function.constant.Constant(0.0)

    a00 = u * v * dx
    a01 = v * p * dx
    a10 = q * u * dx
    a11 = q * p * dx
    # a11 = None

    L0 = zero * f * v * dx
    L1 = g * q * dx

    # Create assembler
    assembler = dolfin.fem.assembling.Assembler([[a00, a01], [a10, a11]],
                                                [L0, L1], [bc])

    # Monolithic blocked
    A0, b0 = assembler.assemble(
        mat_type=dolfin.cpp.fem.Assembler.BlockType.monolithic)
    assert A0.mat().getType() != "nest"
    Anorm0 = A0.mat().norm()
    bnorm0 = b0.vec().norm()

    # Nested (MatNest)
    A1, b1 = assembler.assemble(
        mat_type=dolfin.cpp.fem.Assembler.BlockType.nested)
    assert A1.mat().getType() == "nest"

    bnorm1 = math.sqrt(sum([x.norm()**2 for x in b1.vec().getNestSubVecs()]))
    assert bnorm0 == pytest.approx(bnorm1, 1.0e-12)

    try:
        Anorm1 = 0.0
        nrows, ncols = A1.mat().getNestSize()
        for row in range(nrows):
            for col in range(ncols):
                A_sub = A1.mat().getNestSubMatrix(row, col)
                norm = A_sub.norm()
                Anorm1 += norm * norm
                #A_sub.view()

        # is_rows, is_cols = A1.mat().getNestLocalISs()
        # for is0 in is_rows:
        #     for is1 in is_cols:
        #         A_sub = A1.mat().getLocalSubMatrix(is0, is1)
        #         norm = A_sub.norm()
        #         Anorm1 += norm * norm

        Anorm1 = math.sqrt(Anorm1)
        assert Anorm0 == pytest.approx(Anorm1, 1.0e-12)

    except AttributeError:
        print("Recent petsc4py(-dev) required to get MatNest sub-matrix.")

    # Monolithic version
    E = P0 * P1
    W = dolfin.function.functionspace.FunctionSpace(mesh, E)
    u0, u1 = dolfin.function.argument.TrialFunctions(W)
    v0, v1 = dolfin.function.argument.TestFunctions(W)
    a = u0 * v0 * dx + u1 * v1 * dx + u0 * v1 * dx + u1 * v0 * dx
    L = zero * f * v0 * ufl.dx + g * v1 * dx

    bc = dolfin.fem.dirichletbc.DirichletBC(W.sub(1), u_bc, boundary)
    assembler = dolfin.fem.assembling.Assembler([[a]], [L], [bc])

    A2, b2 = assembler.assemble(
        mat_type=dolfin.cpp.fem.Assembler.BlockType.monolithic)
    assert A2.mat().getType() != "nest"

    Anorm2 = A2.mat().norm()
    bnorm2 = b2.vec().norm()
    assert Anorm0 == pytest.approx(Anorm2, 1.0e-9)
    assert bnorm0 == pytest.approx(bnorm2, 1.0e-9)
