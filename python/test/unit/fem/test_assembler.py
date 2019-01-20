# Copyright (C) 2018-2019 Garth N. Wells
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

    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 4, 8)

    p0, p1 = 1, 2
    P0 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p0)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p1)

    V0 = dolfin.function.functionspace.FunctionSpace(mesh, P0)
    V1 = dolfin.function.functionspace.FunctionSpace(mesh, P1)

    def boundary(x):
        return numpy.logical_or(x[:, 0] < 1.0e-6, x[:, 0] > 1.0 - 1.0e-6)

    u_bc = dolfin.function.Function(V1)
    u_bc.vector().set(50.0)
    u_bc.vector().update_ghosts()
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

    Anorm1 = 0.0
    nrows, ncols = A1.mat().getNestSize()
    for row in range(nrows):
        for col in range(ncols):
            A_sub = A1.mat().getNestSubMatrix(row, col)
            norm = A_sub.norm()
            Anorm1 += norm * norm
    Anorm1 = math.sqrt(Anorm1)

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


def test_assembly_solve_block():
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
    u_bc0.vector().update_ghosts()
    u_bc1 = dolfin.function.Function(V1)
    u_bc1.vector().set(20.0)
    u_bc1.vector().update_ghosts()
    bcs = [
        dolfin.fem.dirichletbc.DirichletBC(V0, u_bc0, boundary),
        dolfin.fem.dirichletbc.DirichletBC(V1, u_bc1, boundary)
    ]

    # Variational problem
    u, p = dolfin.function.argument.TrialFunction(
        V0), dolfin.function.argument.TrialFunction(V1)
    v, q = dolfin.function.argument.TestFunction(
        V0), dolfin.function.argument.TestFunction(V1)
    f = 1.0
    g = -3.0
    zero = dolfin.Function(V0)

    a00 = inner(u, v) * dx
    a01 = zero * inner(p, v) * dx
    a10 = zero * inner(u, q) * dx
    a11 = inner(p, q) * dx
    L0 = inner(f, v) * dx
    L1 = inner(g, q) * dx

    def monitor(ksp, its, rnorm):
        pass
        # print("Norm:", its, rnorm)

    A0 = dolfin.fem.assemble_matrix([[a00, a01], [a10, a11]], bcs,
                                    dolfin.cpp.fem.BlockType.monolithic)
    b0 = dolfin.fem.assemble_vector([L0, L1], [[a00, a01], [a10, a11]], bcs,
                                    dolfin.cpp.fem.BlockType.monolithic)
    A0norm = A0.mat().norm()
    b0norm = b0.vec().norm()
    x0 = A0.mat().createVecLeft()
    ksp = PETSc.KSP()
    ksp.create(mesh.mpi_comm())
    ksp.setOperators(A0.mat())
    ksp.setMonitor(monitor)
    ksp.setType('cg')
    ksp.setTolerances(rtol=1.0e-14)
    ksp.setFromOptions()
    ksp.solve(b0.vec(), x0)
    x0norm = x0.norm()

    # Nested (MatNest)
    A1 = dolfin.fem.assemble_matrix([[a00, a01], [a10, a11]], bcs,
                                    dolfin.cpp.fem.BlockType.nested)
    b1 = dolfin.fem.assemble_vector([L0, L1], [[a00, a01], [a10, a11]], bcs,
                                    dolfin.cpp.fem.BlockType.nested)
    b1norm = b1.vec().norm()
    assert b1norm == pytest.approx(b0norm, 1.0e-12)
    A1norm = 0.0
    nrows, ncols = A1.mat().getNestSize()
    for row in range(nrows):
        for col in range(ncols):
            A_sub = A1.mat().getNestSubMatrix(row, col)
            norm = A_sub.norm()
            A1norm += norm * norm
    A1norm = math.sqrt(A1norm)
    assert A0norm == pytest.approx(A1norm, 1.0e-12)

    x1 = dolfin.la.PETScVector(b1)
    ksp = PETSc.KSP()
    ksp.create(mesh.mpi_comm())
    ksp.setMonitor(monitor)
    ksp.setOperators(A1.mat())
    ksp.setType('cg')
    ksp.setTolerances(rtol=1.0e-12)
    ksp.setFromOptions()
    ksp.solve(b1.vec(), x1.vec())
    x1norm = x1.vec().norm()
    assert x1norm == pytest.approx(x0norm, rel=1.0e-12)

    # Monolithic version
    E = P0 * P1
    W = dolfin.function.functionspace.FunctionSpace(mesh, E)
    u0, u1 = dolfin.function.argument.TrialFunctions(W)
    v0, v1 = dolfin.function.argument.TestFunctions(W)
    a = inner(u0, v0) * dx + inner(u1, v1) * dx
    L = inner(f, v0) * ufl.dx + inner(g, v1) * dx

    V0 = dolfin.function.functionspace.FunctionSpace(mesh, P0)
    V1 = dolfin.function.functionspace.FunctionSpace(mesh, P1)

    u0_bc = dolfin.function.Function(V0)
    u0_bc.vector().set(50.0)
    u0_bc.vector().update_ghosts()

    u1_bc = dolfin.function.Function(V1)
    u1_bc.vector().set(20.0)
    u1_bc.vector().update_ghosts()

    bcs = [
        dolfin.fem.dirichletbc.DirichletBC(W.sub(0), u0_bc, boundary),
        dolfin.fem.dirichletbc.DirichletBC(W.sub(1), u1_bc, boundary)
    ]

    A2 = dolfin.fem.assemble_matrix([[a]], bcs,
                                    dolfin.cpp.fem.BlockType.monolithic)
    b2 = dolfin.fem.assemble_vector([L], [[a]], bcs,
                                    dolfin.cpp.fem.BlockType.monolithic)

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
    ksp.getPC().setType('jacobi')
    ksp.setTolerances(rtol=1.0e-12)
    ksp.setFromOptions()
    ksp.solve(b2.vec(), x2.vec())
    x2norm = x2.vec().norm()
    assert x2norm == pytest.approx(x0norm, 1.0e-10)

    # # Old assembler (reference solution)
    A3, b3 = dolfin.fem.assembling.assemble_system(a, L, bcs)
    x3 = dolfin.cpp.la.PETScVector(b3)
    ksp = PETSc.KSP()
    ksp.create(mesh.mpi_comm())
    ksp.setMonitor(monitor)
    ksp.setOperators(A3.mat())
    ksp.setType('cg')
    ksp.setTolerances(rtol=1.0e-12)
    ksp.setFromOptions()
    ksp.solve(b3.vec(), x3.vec())
    x3norm = x3.vec().norm()
    assert x3norm == pytest.approx(x0norm, 1.0e-10)


@pytest.mark.parametrize("mesh", [
    dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 32, 31),
    dolfin.generation.UnitCubeMesh(dolfin.MPI.comm_world, 3, 7, 8)])
def test_assembly_solve_taylor_hood(mesh):
    """Assemble Stokes problem with Taylor-Hood elements."""

    P2 = dolfin.VectorFunctionSpace(mesh, ("Lagrange", 2))
    P1 = dolfin.FunctionSpace(mesh, ("Lagrange", 1))

    # Define boundary (x = 0)
    def boundary0(x):
        """Define boundary x = 0"""
        return x[:, 0] < 10 * numpy.finfo(float).eps

    def boundary1(x):
        """Define boundary x = 1"""
        return x[:, 0] > (1.0 - 10 * numpy.finfo(float).eps)

    u0 = dolfin.Function(P2)
    u0.vector().set(1.0)
    u0.vector().update_ghosts()
    bc0 = dolfin.DirichletBC(P2, u0, boundary0)
    bc1 = dolfin.DirichletBC(P2, u0, boundary1)

    u, p = dolfin.TrialFunction(P2), dolfin.TrialFunction(P1)
    v, q = dolfin.TestFunction(P2), dolfin.TestFunction(P1)

    a00 = inner(ufl.grad(u), ufl.grad(v)) * dx
    a01 = ufl.inner(p, ufl.div(v)) * dx
    a10 = ufl.inner(ufl.div(u), q) * dx
    a11 = None

    p00 = inner(ufl.grad(u), ufl.grad(v)) * dx
    p01 = None
    p10 = None
    p11 = inner(p, q) * dx

    # FIXME
    # We need zero function for the 'zero' part of L
    p_zero = dolfin.Function(P1)

    f = dolfin.Function(P2)
    L0 = ufl.inner(f, v) * dx
    L1 = ufl.inner(p_zero, q) * dx

    # -- Blocked and nested

    A0 = dolfin.fem.assemble_matrix([[a00, a01], [a10, a11]], [bc0, bc1],
                                    dolfin.cpp.fem.BlockType.nested)
    A0norm = 0.0
    nrows, ncols = A0.mat().getNestSize()
    for row in range(nrows):
        for col in range(ncols):
            A_sub = A0.mat().getNestSubMatrix(row, col)
            if A_sub:
                norm = A_sub.norm()
                A0norm += norm * norm
    A0norm = math.sqrt(A0norm)

    P0 = dolfin.fem.assemble_matrix([[p00, p01], [p10, p11]], [bc0, bc1],
                                    dolfin.cpp.fem.BlockType.nested)
    P0norm = 0.0
    nrows, ncols = P0.mat().getNestSize()
    for row in range(nrows):
        for col in range(ncols):
            P_sub = P0.mat().getNestSubMatrix(row, col)
            if P_sub:
                norm = P_sub.norm()
                P0norm += norm * norm
    P0norm = math.sqrt(P0norm)
    b0 = dolfin.fem.assemble_vector([L0, L1], [[a00, a01], [a10, a11]], [bc0, bc1],
                                    dolfin.cpp.fem.BlockType.nested)
    b0norm = b0.vec().norm()

    ksp = PETSc.KSP()
    ksp.create(mesh.mpi_comm())
    ksp.setOperators(A0.mat(), P0.mat())
    nested_IS = P0.mat().getNestISs()
    ksp.setType("minres")
    pc = ksp.getPC()
    pc.setType("fieldsplit")
    pc.setFieldSplitIS(["u", nested_IS[0][0]], ["p", nested_IS[1][1]])
    ksp_u, ksp_p = pc.getFieldSplitSubKSP()
    ksp_u.setType("preonly")
    ksp_u.getPC().setType('lu')
    ksp_u.getPC().setFactorSolverType('mumps')
    ksp_p.setType("preonly")

    def monitor(ksp, its, rnorm):
        # print("Num it, rnorm:", its, rnorm)
        pass

    ksp.setTolerances(rtol=1.0e-8, max_it=50)
    ksp.setMonitor(monitor)
    ksp.setFromOptions()
    x0 = dolfin.cpp.la.PETScVector(b0)
    ksp.solve(b0.vec(), x0.vec())

    assert ksp.getConvergedReason() > 0

    # -- Blocked and monolithic

    A1 = dolfin.fem.assemble_matrix([[a00, a01], [a10, a11]], [bc0, bc1],
                                    dolfin.cpp.fem.BlockType.monolithic)
    A1norm = A1.mat().norm()
    assert A1norm == pytest.approx(A0norm, 1.0e-12)

    # FIXME
    # P1 = dolfin.fem.assemble_matrix([[p00, p01], [p10, p11]], [bc0, bc1],
    #                                 dolfin.cpp.fem.BlockType.monolithic)
    # P1norm = P1.mat().norm()
    # assert P1norm == pytest.approx(P0norm, 1.0e-12)

    # -- Monolithic

    P2 = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    TH = P2 * P1
    W = dolfin.FunctionSpace(mesh, TH)
    (u, p) = dolfin.TrialFunctions(W)
    (v, q) = dolfin.TestFunctions(W)
    a00 = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    a01 = ufl.inner(p, ufl.div(v)) * dx
    a10 = ufl.inner(ufl.div(u), q) * dx
    a = a00 + a01 + a10

    p00 = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    p11 = ufl.inner(p, q) * dx
    p_form = p00 + p11

    f = dolfin.Function(W.sub(0).collapse())
    p_zero = dolfin.Function(W.sub(1).collapse())
    L0 = inner(f, v) * dx
    L1 = inner(p_zero, q) * dx
    L = L0 + L1

    bc0 = dolfin.DirichletBC(W.sub(0), u0, boundary0)
    bc1 = dolfin.DirichletBC(W.sub(0), u0, boundary1)

    A2 = dolfin.fem.assemble_matrix([[a]], [bc0, bc1], dolfin.cpp.fem.BlockType.monolithic)
    assert A2.mat().norm() == pytest.approx(A0norm, 1.0e-12)

    P2 = dolfin.fem.assemble_matrix([[p_form]], [bc0, bc1], dolfin.cpp.fem.BlockType.monolithic)
    assert P2.mat().norm() == pytest.approx(P0norm, 1.0e-12)

    b2 = dolfin.fem.assemble_vector([L], [[a]], [bc0, bc1], dolfin.cpp.fem.BlockType.monolithic)
    b2norm = b2.vec().norm()
    assert b2norm == pytest.approx(b0norm, 1.0e-12)

    ksp = PETSc.KSP()
    ksp.create(mesh.mpi_comm())
    ksp.setOperators(A2.mat(), P2.mat())
    ksp.setType("minres")
    pc = ksp.getPC()
    pc.setType('lu')
    pc.setFactorSolverType('mumps')

    def monitor(ksp, its, rnorm):
        print("Num it, rnorm:", its, rnorm)
        pass

    ksp.setTolerances(rtol=1.0e-8, max_it=50)
    ksp.setMonitor(monitor)
    ksp.setFromOptions()
    x2 = dolfin.cpp.la.PETScVector(b2)
    ksp.solve(b2.vec(), x2.vec())
    assert ksp.getConvergedReason() > 0

    assert x0.vec().norm() == pytest.approx(x2.vec().norm(), 1e-8)
