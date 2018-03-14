"""Unit tests for assembly"""

# Copyright (C) 2018 Garth N. Wells
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

import pytest
import os
import numpy
import dolfin
import ufl
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
    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 2, 1)

    V0 = dolfin.function.functionspace.FunctionSpace(mesh, "Lagrange", 1)
    V1 = dolfin.function.functionspace.FunctionSpace(mesh, "Lagrange", 1)

    # Define variational problem
    u, p = dolfin.function.argument.TrialFunction(
        V0), dolfin.function.argument.TrialFunction(V1)
    v, q = dolfin.function.argument.TestFunction(
        V0), dolfin.function.argument.TestFunction(V1)
    f = dolfin.function.constant.Constant(-1.0)
    g = dolfin.function.constant.Constant(1.0)

    a00 = u*v*dx
    a01 = v*p * dx
    a10 = q * u * dx
    a11 = q * p * dx
    # a11 = None

    L0 = f*v * dx
    L1 = g*q * dx

    # Define Dirichlet boundary (x = 0 or x = 1)
    def boundary(x):
        return numpy.logical_or(x[:, 0] < 1.0e-6,  x[:, 0] > 1.0 - 1.0e-6)

    u_bc = dolfin.function.constant.Constant(2.0)
    bc = dolfin.fem.dirichletbc.DirichletBC(V1, u_bc, boundary)

    # Create assembler
    assembler = dolfin.fem.assembling.Assembler([[a00, a01], [a10, a11]],
                                                [L0, L1], [bc])

    print("--------------------")

    # Monolithic blocked

    A, b = assembler.assemble(
       mat_type=dolfin.cpp.fem.Assembler.BlockType.monolithic)
    # A.mat().view()
    # b.vec().view()
    norm = A.mat().norm()
    if dolfin.MPI.rank(mesh.mpi_comm()) == 0:
        print("Norm (block, non-nest)", norm)

    dolfin.MPI.barrier(mesh.mpi_comm())
    print("--------------------")
    dolfin.MPI.barrier(mesh.mpi_comm())

    # Nested (MatNest)

    dolfin.MPI.barrier(mesh.mpi_comm())
    A, b = assembler.assemble(
        mat_type=dolfin.cpp.fem.Assembler.BlockType.nested)
    dolfin.MPI.barrier(mesh.mpi_comm())

    # A.mat().view()
    # b.vec().view()

    # try:
    #     IS = A.mat().getNestISs()
    #     print(IS)
    #     IS[0][0].view()
    #     IS[1][0].view()
    #     # IS[1][0].view()
    #     # IS[1][1].view()
    #     # print(A.mat().norm())

    #     print("*** get sub mat")
    #     # A00 = A.mat().getLocalSubMatrix(IS[0][0], IS[1][0])
    #     # A00.view()
    #     # A01 = A.mat().getLocalSubMatrix(IS[0][0], IS[1][1])
    #     print("******")
    #     # A01.view()
    #     # A10 = A.mat().getLocalSubMatrix(IS[0][1], IS[1][0])
    #     # A10.view()
    #     # A11 = A.mat().getLocalSubMatrix(IS[0][1], IS[1][1])
    #     # A11.view()
    # except AttributeError:
    #     print("Recent version of petsc4py required to get MatNest IS.")

    # Monolithic version

    P0 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    E = P0 * P1
    W = dolfin.function.functionspace.FunctionSpace(mesh, E)

    u0, u1 = dolfin.function.argument.TrialFunctions(W)
    v0, v1 = dolfin.function.argument.TestFunctions(W)

    a = u0*v0*dx + u1*v1*dx + u0*v1*dx + u1*v0*dx
    L = f*v0*ufl.dx + g*v1*dx

    dolfin.MPI.barrier(mesh.mpi_comm())
    print("--- Monolithic version --")
    dolfin.MPI.barrier(mesh.mpi_comm())

    bc = dolfin.fem.dirichletbc.DirichletBC(W.sub(1), u_bc, boundary)
    assembler1 = dolfin.fem.assembling.Assembler([[a]], [L], [bc])

    dolfin.MPI.barrier(mesh.mpi_comm())
    A, b = assembler1.assemble(
        mat_type=dolfin.cpp.fem.Assembler.BlockType.monolithic)
    dolfin.MPI.barrier(mesh.mpi_comm())

    # A.mat().view()
    # b.vec().view()
    norm = A.mat().norm()
    if dolfin.MPI.rank(mesh.mpi_comm()) == 0:
        print("Norm (monolithic)", norm)

    # Reference assembler
    # A, b = dolfin.fem.assembling.assemble_system(a, L, bc)
    # A.mat().view()
    # print(A.mat().norm())


def xtest_matrix_assembly_block():
    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 1, 1)

    # P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    # P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    # TH = P2 * P1
    # W = dolfin.function.functionspace.FunctionSpace(mesh, TH)

    P2 = dolfin.function.functionspace.VectorFunctionSpace(mesh, "Lagrange", 1)
    P1 = dolfin.function.functionspace.FunctionSpace(mesh, "Lagrange", 1)

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
    L1 = dolfin.function.constant.Constant(0.0)*q * dx
    # L1 = None

    # Define Dirichlet boundary (x = 0 or x = 1)
    def boundary(x):
        return numpy.logical_or(x[:, 0] < 1.0e-6, x[:, 0] > 1.0 - 1.0e-6)

    u0 = dolfin.function.constant.Constant((2.0, 1.0))
    bc = dolfin.fem.dirichletbc.DirichletBC(P2, u0, boundary)

    assembler = dolfin.fem.assembling.Assembler([[a00, a01], [a10, a11]],
                                                [L0, L1], [bc])
    A, b = assembler.assemble()

    A.mat().view()

    IS = A.mat().getNestISs()
    # print(IS[0][0].view())
    # print(IS[0][1].view())
    print(IS[0][1].view())
    # print(IS[1][1].view())
    # print(A.mat().norm())

    # A00 = A.mat().getLocalSubMatrix(0, 0)
