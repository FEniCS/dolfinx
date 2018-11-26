# Copyright (C) 2018 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly"""

import math

import numpy
import pytest
from petsc4py import PETSc

import dolfin
import ufl
from ufl import dx, inner


def test_assemble_functional():
    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 12, 12)

    M = 1.0 * dx(domain=mesh)
    value = dolfin.fem.assemble(M)
    assert value == pytest.approx(1.0, 1e-12)
    x = dolfin.SpatialCoordinate(mesh)
    M = x[0] * dx(domain=mesh)
    value = dolfin.fem.assemble(M)
    assert value == pytest.approx(0.5, 1e-12)


def test_basic_assembly():
    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 12, 12)
    V = dolfin.FunctionSpace(mesh, ("Lagrange", 1))
    u, v = dolfin.TrialFunction(V), dolfin.TestFunction(V)

    a = 1.0 * inner(u, v) * dx
    L = inner(1.0, v) * dx

    # Initial assembly
    A = dolfin.fem.assemble(a)
    b = dolfin.fem.assemble(L)
    return
    assert isinstance(A, dolfin.cpp.la.PETScMatrix)
    assert isinstance(b, dolfin.cpp.la.PETScVector)


    # Second assembly
    A = dolfin.fem.assemble(A, a)
    b = dolfin.fem.assemble(b, L)
    assert isinstance(A, dolfin.cpp.la.PETScMatrix)
    assert isinstance(b, dolfin.cpp.la.PETScVector)

    # Function as coefficient
    f = dolfin.Function(V)
    a = f * inner(u, v) * dx
    A = dolfin.fem.assemble(a)
    assert isinstance(A, dolfin.cpp.la.PETScMatrix)


def test_matrix_assembly_block():
    """Test assembly of block matrices and vectors into (a) monolithic
    blocked structures, PETSc Nest structures, and monolithic structures.
    """

    # mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 4, 8)
    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 2, 1)

    p0, p1 = 1, 2
    P0 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p0)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p1)

    V0 = dolfin.function.functionspace.FunctionSpace(mesh, P0)
    V1 = dolfin.function.functionspace.FunctionSpace(mesh, P1)

    def boundary(x):
        return numpy.logical_or(x[:, 0] < 1.0e-6, x[:, 0] > 1.0 - 1.0e-6)

    u_bc = dolfin.function.Function(V1)
    u_bc.vector().set(50.0)
    bc = dolfin.fem.dirichletbc.DirichletBC(V1, u_bc, boundary)

    # Define variational problem
    u, p = dolfin.function.argument.TrialFunction(
        V0), dolfin.function.argument.TrialFunction(V1)
    v, q = dolfin.function.argument.TestFunction(
        V0), dolfin.function.argument.TestFunction(V1)
    f = 1.0
    g = -3.0
    zero = dolfin.Function(V0)

    a00 = inner(u, v) * dx
    a01 = inner(p, v) * dx
    a10 = inner(u, q) * dx
    a11 = inner(p, q) * dx
    # a11 = None

    L0 = zero * inner(f, v) * dx
    L1 = inner(g, q) * dx

    a_block = [[a00, a01], [a10, a11]]
    L_block = [L0, L1]

    # Monolithic blocked
    A0 = dolfin.fem.assemble_matrix(a_block, [bc],
                                    dolfin.cpp.fem.BlockType.monolithic)
    b0 = dolfin.fem.assemble_vector(L_block, a_block, [bc],
                                    dolfin.cpp.fem.BlockType.monolithic)
    assert A0.mat().getType() != "nest"
    Anorm0 = A0.mat().norm()
    bnorm0 = b0.vec().norm()

    # Nested (MatNest)
    A1 = dolfin.fem.assemble_matrix(a_block, [bc],
                                    dolfin.cpp.fem.BlockType.nested)
    b1 = dolfin.fem.assemble_vector(L_block, a_block, [bc],
                                    dolfin.cpp.fem.BlockType.nested)
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
                # A_sub.view()

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
    a = inner(u0, v0) * dx + inner(u1, v1) * dx + inner(u0, v1) * dx + inner(
        u1, v0) * dx
    L = zero * inner(f, v0) * ufl.dx + inner(g, v1) * dx

    bc = dolfin.fem.dirichletbc.DirichletBC(W.sub(1), u_bc, boundary)
    A2 = dolfin.fem.assemble_matrix([[a]], [bc],
                                    dolfin.cpp.fem.BlockType.monolithic)
    b2 = dolfin.fem.assemble_vector([L], [[a]], [bc],
                                    dolfin.cpp.fem.BlockType.monolithic)
    assert A2.mat().getType() != "nest"

    Anorm2 = A2.mat().norm()
    bnorm2 = b2.vec().norm()
    assert Anorm0 == pytest.approx(Anorm2, 1.0e-9)
    assert bnorm0 == pytest.approx(bnorm2, 1.0e-9)


def xtest_assembly_solve_block():
    """Solve a two-field mass-matrix like problem with block matrix approaches
    and test that solution is the same.
    """

    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 32, 31)
    p0, p1 = 1, 1
    P0 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p0)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p1)
    V0 = dolfin.function.functionspace.FunctionSpace(mesh, P0)
    V1 = dolfin.function.functionspace.FunctionSpace(mesh, P1)

    def boundary(x):
        return numpy.logical_or(x[:, 0] < 1.0e-6, x[:, 0] > 1.0 - 1.0e-6)

    u_bc0 = dolfin.function.Function(V0)
    u_bc0.vector().set(50.0)

    u_bc1 = dolfin.function.Function(V1)
    u_bc1.vector().set(20.0)

    bc0 = dolfin.fem.dirichletbc.DirichletBC(V0, u_bc0, boundary)
    bc1 = dolfin.fem.dirichletbc.DirichletBC(V1, u_bc1, boundary)

    # Variational problem
    u, p = dolfin.function.argument.TrialFunction(
        V0), dolfin.function.argument.TrialFunction(V1)
    v, q = dolfin.function.argument.TestFunction(
        V0), dolfin.function.argument.TestFunction(V1)
    f = 1.0
    g = -3.0
    zero = 0.0

    a00 = inner(u, v) * dx
    a01 = zero * inner(p, v) * dx
    a10 = zero * inner(u, q) * dx
    a11 = inner(p, q) * dx
    L0 = inner(f, v) * dx
    L1 = inner(g, q) * dx

    def monitor(ksp, its, rnorm):
        pass
        # print("Norm:", its, rnorm)

    # Create assembler
    assembler = dolfin.fem.Assembler([[a00, a01], [a10, a11]], [L0, L1],
                                     [bc0, bc1])

    # Monolithic blocked
    A0, b0 = assembler.assemble(
        mat_type=dolfin.cpp.fem.Assembler.BlockType.monolithic)
    A0norm = A0.mat().norm()
    b0norm = b0.vec().norm()
    x0 = A0.mat().createVecLeft()
    ksp = PETSc.KSP()
    ksp.create(mesh.mpi_comm())
    ksp.setOperators(A0.mat())
    ksp.setTolerances(rtol=1.0e-12)
    ksp.setMonitor(monitor)
    ksp.setType('cg')
    ksp.setFromOptions()
    # ksp.view()
    ksp.solve(b0.vec(), x0)
    x0norm = x0.norm()

    # Nested (MatNest)
    A1, b1 = assembler.assemble(
        mat_type=dolfin.cpp.fem.Assembler.BlockType.nested)
    b1norm = b1.vec().norm()
    assert b1norm == pytest.approx(b0norm, 1.0e-12)

    x1 = dolfin.la.PETScVector(b1)
    ksp = PETSc.KSP()
    ksp.create(mesh.mpi_comm())
    ksp.setMonitor(monitor)
    ksp.setTolerances(rtol=1.0e-12)
    ksp.setOperators(A1.mat())
    ksp.setType('cg')
    ksp.setFromOptions()
    # ksp.view()
    ksp.solve(b1.vec(), x1.vec())
    x1norm = x1.vec().norm()
    assert x1norm == pytest.approx(x0norm, rel=1.0e-10)

    # Monolithic version
    E = P0 * P1
    W = dolfin.function.functionspace.FunctionSpace(mesh, E)
    u0, u1 = dolfin.function.argument.TrialFunctions(W)
    v0, v1 = dolfin.function.argument.TestFunctions(W)
    a = inner(u0, v0) * dx + inner(u1, v1) * dx
    L = inner(f, v0) * ufl.dx + inner(g, v1) * dx

    u1_bc = dolfin.function.Function(P0)
    u1_bc.vector().set(50.0)

    u2_bc = dolfin.function.Function(P1)
    u2_bc.vector().set(20.0)

    bcs = []
    bcs.append(dolfin.fem.dirichletbc.DirichletBC(W.sub(0), u1_bc, boundary))
    bcs.append(dolfin.fem.dirichletbc.DirichletBC(W.sub(1), u2_bc, boundary))

    assembler = dolfin.fem.assembler.Assembler([[a]], [L], bcs)

    A2, b2 = assembler.assemble(
        mat_type=dolfin.cpp.fem.Assembler.BlockType.monolithic)
    A2norm = A2.mat().norm()
    b2norm = b2.vec().norm()
    assert A2norm == pytest.approx(A0norm, 1.0e-12)
    assert b2norm == pytest.approx(b0norm, 1.0e-12)

    x2 = dolfin.cpp.la.PETScVector(b2)
    ksp = PETSc.KSP()
    ksp.create(mesh.mpi_comm())
    ksp.setMonitor(monitor)
    ksp.setOperators(A2.mat())
    ksp.setType('cg')
    ksp.setTolerances(rtol=1.0e-9)
    ksp.setFromOptions()
    # ksp.view()
    ksp.solve(b2.vec(), x2.vec())
    x2norm = x2.vec().norm()
    assert x2norm == pytest.approx(x0norm, 1.0e-10)

    # Old assembler (reference)
    A3, b3 = dolfin.fem.assembling.assemble_system(a, L, bcs)
    x3 = dolfin.cpp.la.PETScVector(b3)
    ksp = PETSc.KSP()
    ksp.create(mesh.mpi_comm())
    ksp.setMonitor(monitor)
    ksp.setOperators(A3.mat())
    ksp.setType('cg')
    ksp.setTolerances(rtol=1.0e-9)
    ksp.setFromOptions()
    # ksp.view()
    ksp.solve(b3.vec(), x3.vec())
    x3norm = x3.vec().norm()
    assert x3norm == pytest.approx(x0norm, 1.0e-10)
