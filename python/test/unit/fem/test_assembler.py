"""Unit tests for assembly"""

# Copyright (C) 2018 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import math
import os

import numpy
import pytest

import dolfin
import ufl
from dolfin_utils.test import skip_in_parallel
from ufl import dx


def xtest_initialisation():
    "Test intialisation of the assembler"
    mesh = dolfin.generation.UnitCubeMesh(dolfin.MPI.comm_world, 4, 4, 4)
    V = dolfin.function.functionspace.FunctionSpace(mesh, "Lagrange", 1)
    v = dolfin.function.argument.TestFunction(V)
    u = dolfin.function.argument.TrialFunction(V)
    f = dolfin.function.constant.Constant(0.0)
    a = v * u * dx
    L = v * f * dx

    assembler = dolfin.fem.assembling.Assembler(a, L)
    assembler = dolfin.fem.assembling.Assembler([[a, a], [a, a]], [L, L])

    # TODO: test that exceptions are raised for incorrect input
    # arguments


def xtest_matrix_assembly():
    "Test basic assembly without Dirichlet boundary conditions"
    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 8, 8)
    V = dolfin.function.functionspace.FunctionSpace(mesh, "Lagrange", 1)
    v = dolfin.function.argument.TestFunction(V)
    u = dolfin.function.argument.TrialFunction(V)
    f = dolfin.function.constant.Constant(1.0)
    a = v * u * dx
    L = v * f * dx

    assembler = dolfin.fem.assembling.Assembler(a, L)
    A, b = assembler.assemble()

    # Old assembler for reference (requires petsc4py)
    B = dolfin.cpp.la.PETScMatrix(mesh.mpi_comm())
    c = dolfin.cpp.la.PETScVector(mesh.mpi_comm())
    ass0 = dolfin.fem.assembling.SystemAssembler(a, L)
    ass0.assemble(B, c)

    assert pytest.approx(0.0, 1.0e-17) == (A.mat() - B.mat()).norm()
    assert pytest.approx(0.0, 1.0e-17) == (b.vec() - c.vec()).norm()

    # b.vec().view()
    # c.vec().view()

    # A.mat().view()
    # B.mat().view()
    # print(c.vec().getArray())


def xtest_matrix_assembly_bc():
    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 2, 1)
    V = dolfin.function.functionspace.FunctionSpace(mesh, "Lagrange", 1)
    v = dolfin.function.argument.TestFunction(V)
    u = dolfin.function.argument.TrialFunction(V)
    f = dolfin.function.constant.Constant(1.0)
    a = v * u * dx
    L = v * f * dx

    # Define Dirichlet boundary (x = 0 or x = 1)
    def boundary(x):
        return numpy.logical_or(x[:, 0] < 1.0e-6, x[:, 0] > 1.0 - 1.0e-6)

    u0 = dolfin.function.constant.Constant(2.0)
    bc = dolfin.fem.dirichletbc.DirichletBC(V, u0, boundary)

    assembler = dolfin.fem.assembling.Assembler(a, L, [bc])
    A, b = assembler.assemble()

    # Old assembler for reference (requires petsc4py)
    B = dolfin.cpp.la.PETScMatrix(mesh.mpi_comm())
    c = dolfin.cpp.la.PETScVector(mesh.mpi_comm())
    ass0 = dolfin.fem.assembling.SystemAssembler(a, L, [bc])
    ass0.assemble(B, c)

    b.vec().view()
    c.vec().view()
    # A.mat().view()
    # B.mat().view()


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

    u_bc = dolfin.function.constant.Constant(1.0)
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
                                                [L0, L1], [])

    # Monolithic blocked
    A0, b0 = assembler.assemble(
        mat_type=dolfin.cpp.fem.Assembler.BlockType.monolithic)
    Anorm0 = A0.mat().norm()
    #bnorm0 = b0.vec().norm()
    # if dolfin.MPI.rank(mesh.mpi_comm()) == 0:
    #     print("Matrix Norm (block, non-nest)", Anorm0)
    #     print("Vector Norm (block, non-nest)", bnorm0)

    # Nested (MatNest)
    dolfin.MPI.barrier(mesh.mpi_comm())
    A1, b1 = assembler.assemble(
        mat_type=dolfin.cpp.fem.Assembler.BlockType.nested)

    return
    is_rows, is_cols = A1.mat().getNestISs()
    Anorm1 = 0.0
    for is0 in is_rows:
        for is1 in is_cols:
            A_sub = A1.mat().getLocalSubMatrix(is0, is1)
            norm = A_sub.norm()
            Anorm1 += norm * norm
    Anorm1 = math.sqrt(Anorm1)
    bnorm1 = math.sqrt(sum([x.norm()**2 for x in b1.vec().getNestSubVecs()]))

    assert Anorm0 == pytest.approx(Anorm1, 1.0e-12)
    assert bnorm0 == pytest.approx(bnorm1, 1.0e-12)

    # Monolithic version
    E = P0 * P1
    W = dolfin.function.functionspace.FunctionSpace(mesh, E)
    u0, u1 = dolfin.function.argument.TrialFunctions(W)
    v0, v1 = dolfin.function.argument.TestFunctions(W)
    a = u0 * v0 * dx + u1 * v1 * dx + u0 * v1 * dx + u1 * v0 * dx
    L = zero * f * v0 * ufl.dx + g * v1 * dx

    bc = dolfin.fem.dirichletbc.DirichletBC(W.sub(1), u_bc, boundary)
    assembler = dolfin.fem.assembling.Assembler([[a]], [L], [])

    A2, b2 = assembler.assemble(
        mat_type=dolfin.cpp.fem.Assembler.BlockType.monolithic)
    dolfin.MPI.barrier(mesh.mpi_comm())
    Anorm2 = A2.mat().norm()
    bnorm2 = b2.vec().norm()
    # if dolfin.MPI.rank(mesh.mpi_comm()) == 0:
    #     print("Matrix norm (monolithic)", Anorm2)
    #     print("Vector Norm (monolithic)", bnorm2)

    assert Anorm0 == pytest.approx(Anorm2, 1.0e-9)
    assert bnorm0 == pytest.approx(bnorm2, 1.0e-9)


def xtest_matrix_assembly_block():
    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 1, 1)

    # P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    # P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    # TH = P2 * P1
    # W = dolfin.function.functionspace.FunctionSpace(mesh, TH)

    P2 = dolfin.function.functionspace.VectorFunctionSpace(mesh, "Lagrange", 2)
    P1 = dolfin.function.functionspace.FunctionSpace(mesh, "Lagrange", 2)

    # Define variational problem
    u, p = dolfin.function.argument.TrialFunction(
        P2), dolfin.function.argument.TrialFunction(P1)
    v, q = dolfin.function.argument.TestFunction(
        P2), dolfin.function.argument.TestFunction(P1)
    #(u, p) = dolfin.function.argument.TrialFunctions(W)
    #(v, q) = dolfin.function.argument.TestFunctions(W)
    f = dolfin.function.constant.Constant((0, 0))

    # a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - ufl.div(v)*p + q*ufl.div(u))*dx
    # L = ufl.inner(f, v)*dx

    a00 = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    a01 = -ufl.div(v) * p * dx
    a10 = q * ufl.div(u) * dx
    a11 = dolfin.function.constant.Constant(0.0) * q * p * dx
    # a11 = None

    L0 = ufl.inner(f, v) * dx
    L1 = dolfin.function.constant.Constant(0.0) * q * dx

    # L1 = None

    # Define Dirichlet boundary (x = 0 or x = 1)
    def boundary(x):
        return numpy.logical_or(x[:, 0] < 1.0e-6, x[:, 0] > 1.0 - 1.0e-6)

    u0 = dolfin.function.constant.Constant((2.0, 1.0))
    bc = dolfin.fem.dirichletbc.DirichletBC(P2, u0, boundary)

    assembler = dolfin.fem.assembling.Assembler([[a00, a01], [a10, a11]],
                                                [L0, L1], [])
    A, b = assembler.assemble()

    A.mat().view()

    IS = A.mat().getNestISs()
    # print(IS[0][0].view())
    # print(IS[0][1].view())
    print(IS[0][1].view())
    # print(IS[1][1].view())
    # print(A.mat().norm())

    # A00 = A.mat().getLocalSubMatrix(0, 0)
