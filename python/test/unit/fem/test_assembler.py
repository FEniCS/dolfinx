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

import dolfinx
import ufl
from dolfinx import constant, functionspace
from dolfinx.specialfunctions import SpatialCoordinate
from ufl import derivative, ds, dx, inner


def nest_matrix_norm(A):
    """Return norm of a MatNest matrix"""
    assert A.getType() == "nest"
    norm = 0.0
    nrows, ncols = A.getNestSize()
    for row in range(nrows):
        for col in range(ncols):
            A_sub = A.getNestSubMatrix(row, col)
            if A_sub:
                _norm = A_sub.norm()
                norm += _norm * _norm
    return math.sqrt(norm)


def test_assemble_functional():
    mesh = dolfinx.generation.UnitSquareMesh(dolfinx.MPI.comm_world, 12, 12)
    M = 1.0 * dx(domain=mesh)
    value = dolfinx.fem.assemble_scalar(M)
    value = dolfinx.MPI.sum(mesh.mpi_comm(), value)
    assert value == pytest.approx(1.0, 1e-12)
    x = SpatialCoordinate(mesh)
    M = x[0] * dx(domain=mesh)
    value = dolfinx.fem.assemble_scalar(M)
    value = dolfinx.MPI.sum(mesh.mpi_comm(), value)
    assert value == pytest.approx(0.5, 1e-12)


def test_assemble_derivatives():
    """This test checks the original_coefficient_positions, which may change
    under differentiation (some coefficients and constants are
    eliminated)"""
    mesh = dolfinx.generation.UnitSquareMesh(dolfinx.MPI.comm_world, 12, 12)
    Q = dolfinx.FunctionSpace(mesh, ("Lagrange", 1))
    u = dolfinx.Function(Q)
    v = ufl.TestFunction(Q)
    du = ufl.TrialFunction(Q)
    b = dolfinx.Function(Q)
    c1 = constant.Constant(mesh, [[1.0, 0.0], [3.0, 4.0]])
    c2 = constant.Constant(mesh, 2.0)

    with b.vector.localForm() as b_local:
        b_local.set(2.0)

    # derivative eliminates 'u' and 'c1'
    L = ufl.inner(c1, c1) * v * dx + c2 * b * inner(u, v) * dx
    a = derivative(L, u, du)
    A1 = dolfinx.fem.assemble_matrix(a)
    A1.assemble()

    a = c2 * b * inner(du, v) * dx
    A2 = dolfinx.fem.assemble_matrix(a)
    A2.assemble()

    assert (A1 - A2).norm() == pytest.approx(0.0, rel=1e-12, abs=1e-12)


def test_basic_assembly():
    mesh = dolfinx.generation.UnitSquareMesh(dolfinx.MPI.comm_world, 12, 12)
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    f = dolfinx.Function(V)
    with f.vector.localForm() as f_local:
        f_local.set(10.0)
    a = inner(f * u, v) * dx + inner(u, v) * ds
    L = inner(f, v) * dx + inner(2.0, v) * ds

    # Initial assembly
    A = dolfinx.fem.assemble_matrix(a)
    A.assemble()
    assert isinstance(A, PETSc.Mat)
    b = dolfinx.fem.assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert isinstance(b, PETSc.Vec)

    # Second assembly
    normA = A.norm()
    A.zeroEntries()
    A = dolfinx.fem.assemble_matrix(A, a)
    A.assemble()
    assert isinstance(A, PETSc.Mat)
    assert normA == pytest.approx(A.norm())
    normb = b.norm()
    with b.localForm() as b_local:
        b_local.set(0.0)
    b = dolfinx.fem.assemble_vector(b, L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert isinstance(b, PETSc.Vec)
    assert normb == pytest.approx(b.norm())

    # Vector re-assembly - no zeroing (but need to zero ghost entries)
    with b.localForm() as b_local:
        b_local.array[b.local_size:] = 0.0
    dolfinx.fem.assemble_vector(b, L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert 2.0 * normb == pytest.approx(b.norm())

    # Matrix re-assembly (no zeroing)
    dolfinx.fem.assemble_matrix(A, a)
    A.assemble()
    assert 2.0 * normA == pytest.approx(A.norm())


def test_assembly_bcs():
    mesh = dolfinx.generation.UnitSquareMesh(dolfinx.MPI.comm_world, 12, 12)
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = inner(u, v) * dx + inner(u, v) * ds
    L = inner(1.0, v) * dx

    def boundary(x):
        return numpy.logical_or(x[0] < 1.0e-6, x[0] > 1.0 - 1.0e-6)

    u_bc = dolfinx.function.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(1.0)
    bc = dolfinx.fem.dirichletbc.DirichletBC(V, u_bc, boundary)

    # Assemble and apply 'global' lifting of bcs
    A = dolfinx.fem.assemble_matrix(a)
    A.assemble()
    b = dolfinx.fem.assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    g = b.duplicate()
    with g.localForm() as g_local:
        g_local.set(0.0)
    dolfinx.fem.set_bc(g, [bc])
    f = b - A * g
    dolfinx.fem.set_bc(f, [bc])

    # Assemble vector and apply lifting of bcs during assembly
    b = dolfinx.fem.assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    b_bc = dolfinx.fem.assemble_vector(L)
    dolfinx.fem.apply_lifting(b_bc, [a], [[bc]])
    b_bc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b_bc, [bc])

    assert (f - b_bc).norm() == pytest.approx(0.0, rel=1e-12, abs=1e-12)


def test_matrix_assembly_block():
    """Test assembly of block matrices and vectors into (a) monolithic
    blocked structures, PETSc Nest structures, and monolithic structures.
    """
    mesh = dolfinx.generation.UnitSquareMesh(dolfinx.MPI.comm_world, 4, 8)

    p0, p1 = 1, 2
    P0 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p0)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p1)

    V0 = dolfinx.function.functionspace.FunctionSpace(mesh, P0)
    V1 = dolfinx.function.functionspace.FunctionSpace(mesh, P1)

    def boundary(x):
        return numpy.logical_or(x[0] < 1.0e-6, x[0] > 1.0 - 1.0e-6)

    u_bc = dolfinx.function.Function(V1)
    with u_bc.vector.localForm() as u_local:
        u_local.set(50.0)
    bc = dolfinx.fem.dirichletbc.DirichletBC(V1, u_bc, boundary)

    # Define variational problem
    u, p = ufl.TrialFunction(V0), ufl.TrialFunction(V1)
    v, q = ufl.TestFunction(V0), ufl.TestFunction(V1)
    f = 1.0
    g = -3.0
    zero = dolfinx.Function(V0)

    a00 = inner(u, v) * dx
    a01 = inner(p, v) * dx
    a10 = inner(u, q) * dx
    a11 = inner(p, q) * dx

    L0 = zero * inner(f, v) * dx
    L1 = inner(g, q) * dx

    a_block = [[a00, a01], [a10, a11]]
    L_block = [L0, L1]

    # Monolithic blocked
    A0 = dolfinx.fem.assemble_matrix_block(a_block, [bc])
    A0.assemble()
    b0 = dolfinx.fem.assemble_vector_block(L_block, a_block, [bc])
    assert A0.getType() != "nest"
    Anorm0 = A0.norm()
    bnorm0 = b0.norm()

    # Nested (MatNest)
    A1 = dolfinx.fem.assemble_matrix_nest(a_block, [bc])
    A1.assemble()
    Anorm1 = nest_matrix_norm(A1)
    assert Anorm0 == pytest.approx(Anorm1, 1.0e-12)

    b1 = dolfinx.fem.assemble.assemble_vector_nest(L_block)
    dolfinx.fem.assemble.apply_lifting_nest(b1, a_block, [bc])
    for b_sub in b1.getNestSubVecs():
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    bcs0 = dolfinx.cpp.fem.bcs_rows(dolfinx.fem.assemble._create_cpp_form(L_block), [bc])
    dolfinx.fem.assemble.set_bc_nest(b1, bcs0)
    b1.assemble()

    bnorm1 = math.sqrt(sum([x.norm()**2 for x in b1.getNestSubVecs()]))
    assert bnorm0 == pytest.approx(bnorm1, 1.0e-12)

    # Monolithic version
    E = P0 * P1
    W = dolfinx.function.functionspace.FunctionSpace(mesh, E)
    u0, u1 = ufl.TrialFunctions(W)
    v0, v1 = ufl.TestFunctions(W)
    a = inner(u0, v0) * dx + inner(u1, v1) * dx + inner(u0, v1) * dx + inner(
        u1, v0) * dx
    L = zero * inner(f, v0) * ufl.dx + inner(g, v1) * dx

    bc = dolfinx.fem.dirichletbc.DirichletBC(W.sub(1), u_bc, boundary)
    A2 = dolfinx.fem.assemble_matrix(a, [bc])
    A2.assemble()
    b2 = dolfinx.fem.assemble_vector(L)
    dolfinx.fem.apply_lifting(b2, [a], [[bc]])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b2, [bc])
    assert A2.getType() != "nest"
    assert A2.norm() == pytest.approx(Anorm0, 1.0e-9)
    assert b2.norm() == pytest.approx(bnorm0, 1.0e-9)


def test_assembly_solve_block():
    """Solve a two-field mass-matrix like problem with block matrix approaches
    and test that solution is the same.
    """
    mesh = dolfinx.generation.UnitSquareMesh(dolfinx.MPI.comm_world, 32, 31)
    P = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    V0 = dolfinx.function.functionspace.FunctionSpace(mesh, P)
    V1 = V0.clone()

    def boundary(x):
        return numpy.logical_or(x[0] < 1.0e-6, x[0] > 1.0 - 1.0e-6)

    u_bc0 = dolfinx.function.Function(V0)
    u_bc0.vector.set(50.0)
    u_bc0.vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    u_bc1 = dolfinx.function.Function(V1)
    u_bc1.vector.set(20.0)
    u_bc1.vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    bcs = [
        dolfinx.fem.dirichletbc.DirichletBC(V0, u_bc0, boundary),
        dolfinx.fem.dirichletbc.DirichletBC(V1, u_bc1, boundary)
    ]

    # Variational problem
    u, p = ufl.TrialFunction(V0), ufl.TrialFunction(V1)
    v, q = ufl.TestFunction(V0), ufl.TestFunction(V1)
    f = 1.0
    g = -3.0
    zero = dolfinx.Function(V0)

    a00 = inner(u, v) * dx
    a01 = zero * inner(p, v) * dx
    a10 = zero * inner(u, q) * dx
    a11 = inner(p, q) * dx
    L0 = inner(f, v) * dx
    L1 = inner(g, q) * dx

    def monitor(ksp, its, rnorm):
        pass
        # print("Norm:", its, rnorm)

    A0 = dolfinx.fem.assemble_matrix_block([[a00, a01], [a10, a11]], bcs)
    b0 = dolfinx.fem.assemble_vector_block([L0, L1], [[a00, a01], [a10, a11]],
                                          bcs)
    A0.assemble()
    A0norm = A0.norm()
    b0norm = b0.norm()
    x0 = A0.createVecLeft()
    ksp = PETSc.KSP()
    ksp.create(mesh.mpi_comm())
    ksp.setOperators(A0)
    ksp.setMonitor(monitor)
    ksp.setType('cg')
    ksp.setTolerances(rtol=1.0e-14)
    ksp.setFromOptions()
    ksp.solve(b0, x0)
    x0norm = x0.norm()

    # Nested (MatNest)
    A1 = dolfinx.fem.assemble_matrix_nest([[a00, a01], [a10, a11]], bcs)
    A1.assemble()
    b1 = dolfinx.fem.assemble.assemble_vector_nest([L0, L1])
    dolfinx.fem.assemble.apply_lifting_nest(b1, [[a00, a01], [a10, a11]], bcs)
    for b_sub in b1.getNestSubVecs():
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    bcs0 = dolfinx.cpp.fem.bcs_rows(dolfinx.fem.assemble._create_cpp_form([L0, L1]), bcs)
    dolfinx.fem.assemble.set_bc_nest(b1, bcs0)
    b1.assemble()

    b1norm = b1.norm()
    assert b1norm == pytest.approx(b0norm, 1.0e-12)
    A1norm = nest_matrix_norm(A1)
    assert A0norm == pytest.approx(A1norm, 1.0e-12)

    x1 = b1.copy()
    ksp = PETSc.KSP()
    ksp.create(mesh.mpi_comm())
    ksp.setMonitor(monitor)
    ksp.setOperators(A1)
    ksp.setType('cg')
    ksp.setTolerances(rtol=1.0e-12)
    ksp.setFromOptions()
    ksp.solve(b1, x1)
    x1norm = x1.norm()
    assert x1norm == pytest.approx(x0norm, rel=1.0e-12)

    # Monolithic version
    E = P * P
    W = dolfinx.function.functionspace.FunctionSpace(mesh, E)
    u0, u1 = ufl.TrialFunctions(W)
    v0, v1 = ufl.TestFunctions(W)
    a = inner(u0, v0) * dx + inner(u1, v1) * dx
    L = inner(f, v0) * ufl.dx + inner(g, v1) * dx

    u0_bc = dolfinx.function.Function(V0)
    u0_bc.vector.set(50.0)
    u0_bc.vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    u1_bc = dolfinx.function.Function(V1)
    u1_bc.vector.set(20.0)
    u1_bc.vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    bcs = [
        dolfinx.fem.dirichletbc.DirichletBC(W.sub(0), u0_bc, boundary),
        dolfinx.fem.dirichletbc.DirichletBC(W.sub(1), u1_bc, boundary)
    ]

    A2 = dolfinx.fem.assemble_matrix(a, bcs)
    A2.assemble()
    b2 = dolfinx.fem.assemble_vector(L)
    dolfinx.fem.apply_lifting(b2, [a], [bcs])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b2, bcs)
    A2norm = A2.norm()
    b2norm = b2.norm()
    assert A2norm == pytest.approx(A0norm, 1.0e-12)
    assert b2norm == pytest.approx(b0norm, 1.0e-12)

    x2 = b2.copy()
    ksp = PETSc.KSP()
    ksp.create(mesh.mpi_comm())
    ksp.setMonitor(monitor)
    ksp.setOperators(A2)
    ksp.setType('cg')
    ksp.getPC().setType('jacobi')
    ksp.setTolerances(rtol=1.0e-12)
    ksp.setFromOptions()
    ksp.solve(b2, x2)
    x2norm = x2.norm()
    assert x2norm == pytest.approx(x0norm, 1.0e-10)


@pytest.mark.parametrize("mesh", [
    dolfinx.generation.UnitSquareMesh(dolfinx.MPI.comm_world, 12, 11),
    dolfinx.generation.UnitCubeMesh(dolfinx.MPI.comm_world, 3, 7, 8)
])
def test_assembly_solve_taylor_hood(mesh):
    """Assemble Stokes problem with Taylor-Hood elements and solve."""
    P2 = functionspace.VectorFunctionSpace(mesh, ("Lagrange", 2))
    P1 = functionspace.FunctionSpace(mesh, ("Lagrange", 1))

    def boundary0(x):
        """Define boundary x = 0"""
        return x[0] < 10 * numpy.finfo(float).eps

    def boundary1(x):
        """Define boundary x = 1"""
        return x[0] > (1.0 - 10 * numpy.finfo(float).eps)

    u0 = dolfinx.Function(P2)
    u0.vector.set(1.0)
    u0.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    bc0 = dolfinx.DirichletBC(P2, u0, boundary0)
    bc1 = dolfinx.DirichletBC(P2, u0, boundary1)

    u, p = ufl.TrialFunction(P2), ufl.TrialFunction(P1)
    v, q = ufl.TestFunction(P2), ufl.TestFunction(P1)

    a00 = inner(ufl.grad(u), ufl.grad(v)) * dx
    a01 = ufl.inner(p, ufl.div(v)) * dx
    a10 = ufl.inner(ufl.div(u), q) * dx
    a11 = None

    p00 = a00
    p01, p10 = None, None
    p11 = inner(p, q) * dx

    # FIXME
    # We need zero function for the 'zero' part of L
    p_zero = dolfinx.Function(P1)
    f = dolfinx.Function(P2)
    L0 = ufl.inner(f, v) * dx
    L1 = ufl.inner(p_zero, q) * dx

    # -- Blocked (nested)

    A0 = dolfinx.fem.assemble_matrix_nest([[a00, a01], [a10, a11]], [bc0, bc1])
    A0.assemble()
    A0norm = nest_matrix_norm(A0)
    P0 = dolfinx.fem.assemble_matrix_nest([[p00, p01], [p10, p11]], [bc0, bc1])
    P0.assemble()
    P0norm = nest_matrix_norm(P0)
    b0 = dolfinx.fem.assemble.assemble_vector_nest([L0, L1])
    dolfinx.fem.assemble.apply_lifting_nest(b0, [[a00, a01], [a10, a11]], [bc0, bc1])
    for b_sub in b0.getNestSubVecs():
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    bcs0 = dolfinx.cpp.fem.bcs_rows(dolfinx.fem.assemble._create_cpp_form([L0, L1]), [bc0, bc1])
    dolfinx.fem.assemble.set_bc_nest(b0, bcs0)
    b0.assemble()
    b0norm = b0.norm()

    ksp = PETSc.KSP()
    ksp.create(mesh.mpi_comm())
    ksp.setOperators(A0, P0)
    nested_IS = P0.getNestISs()
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
    x0 = b0.copy()
    ksp.solve(b0, x0)
    assert ksp.getConvergedReason() > 0

    # -- Blocked (monolithic)

    A1 = dolfinx.fem.assemble_matrix_block([[a00, a01], [a10, a11]], [bc0, bc1])
    A1.assemble()
    assert A1.norm() == pytest.approx(A0norm, 1.0e-12)
    P1 = dolfinx.fem.assemble_matrix_block([[p00, p01], [p10, p11]], [bc0, bc1])
    P1.assemble()
    assert P1.norm() == pytest.approx(P0norm, 1.0e-12)

    b1 = dolfinx.fem.assemble_vector_block([L0, L1], [[a00, a01], [a10, a11]],
                                          [bc0, bc1])

    assert b1.norm() == pytest.approx(b0norm, 1.0e-12)

    ksp = PETSc.KSP()
    ksp.create(mesh.mpi_comm())
    ksp.setOperators(A1, P1)
    ksp.setType("minres")
    pc = ksp.getPC()
    pc.setType('lu')
    pc.setFactorSolverType('mumps')
    ksp.setTolerances(rtol=1.0e-8, max_it=50)
    ksp.setFromOptions()
    x1 = A1.createVecRight()
    ksp.solve(b1, x1)
    assert ksp.getConvergedReason() > 0
    assert x1.norm() == pytest.approx(x0.norm(), 1e-8)

    # -- Monolithic

    P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    TH = P2 * P1
    W = dolfinx.FunctionSpace(mesh, TH)
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    a00 = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    a01 = ufl.inner(p, ufl.div(v)) * dx
    a10 = ufl.inner(ufl.div(u), q) * dx
    a = a00 + a01 + a10

    p00 = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    p11 = ufl.inner(p, q) * dx
    p_form = p00 + p11

    f = dolfinx.Function(W.sub(0).collapse())
    p_zero = dolfinx.Function(W.sub(1).collapse())
    L0 = inner(f, v) * dx
    L1 = inner(p_zero, q) * dx
    L = L0 + L1

    bc0 = dolfinx.DirichletBC(W.sub(0), u0, boundary0)
    bc1 = dolfinx.DirichletBC(W.sub(0), u0, boundary1)

    A2 = dolfinx.fem.assemble_matrix(a, [bc0, bc1])
    A2.assemble()
    assert A2.norm() == pytest.approx(A0norm, 1.0e-12)
    P2 = dolfinx.fem.assemble_matrix(p_form, [bc0, bc1])
    P2.assemble()
    assert P2.norm() == pytest.approx(P0norm, 1.0e-12)

    b2 = dolfinx.fem.assemble_vector(L)
    dolfinx.fem.apply_lifting(b2, [a], [[bc0, bc1]])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b2, [bc0, bc1])
    b2norm = b2.norm()
    assert b2norm == pytest.approx(b0norm, 1.0e-12)

    ksp = PETSc.KSP()
    ksp.create(mesh.mpi_comm())
    ksp.setOperators(A2, P2)
    ksp.setType("minres")
    pc = ksp.getPC()
    pc.setType('lu')
    pc.setFactorSolverType('mumps')

    def monitor(ksp, its, rnorm):
        # print("Num it, rnorm:", its, rnorm)
        pass

    ksp.setTolerances(rtol=1.0e-8, max_it=50)
    ksp.setMonitor(monitor)
    ksp.setFromOptions()
    x2 = A2.createVecRight()
    ksp.solve(b2, x2)
    assert ksp.getConvergedReason() > 0
    assert x0.norm() == pytest.approx(x2.norm(), 1e-8)


def test_basic_interior_facet_assembly():
    ghost_mode = dolfinx.cpp.mesh.GhostMode.none
    if (dolfinx.MPI.size(dolfinx.MPI.comm_world) > 1):
        ghost_mode = dolfinx.cpp.mesh.GhostMode.shared_facet

    mesh = dolfinx.RectangleMesh(dolfinx.MPI.comm_world, [numpy.array([0.0, 0.0, 0.0]),
                                                        numpy.array([1.0, 1.0, 0.0])], [5, 5],
                                cell_type=dolfinx.cpp.mesh.CellType.triangle,
                                ghost_mode=ghost_mode)

    V = functionspace.FunctionSpace(mesh, ("DG", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    a = ufl.inner(ufl.avg(u), ufl.avg(v)) * ufl.dS

    A = dolfinx.fem.assemble_matrix(a)
    A.assemble()
    assert isinstance(A, PETSc.Mat)

    L = ufl.conj(ufl.avg(v)) * ufl.dS

    b = dolfinx.fem.assemble_vector(L)
    b.assemble()
    assert isinstance(b, PETSc.Vec)


def test_basic_assembly_constant():
    """Tests assembly with Constant

    The following test should be sensitive to order of flattening the
    matrix-valued constant.

    """
    mesh = dolfinx.generation.UnitSquareMesh(dolfinx.MPI.comm_world, 5, 5)
    V = functionspace.FunctionSpace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    c = constant.Constant(mesh, [[1.0, 2.0], [5.0, 3.0]])

    a = inner(c[1, 0] * u, v) * dx + inner(c[1, 0] * u, v) * ds
    L = inner(c[1, 0], v) * dx + inner(c[1, 0], v) * ds

    # Initial assembly
    A1 = dolfinx.fem.assemble_matrix(a)
    A1.assemble()

    b1 = dolfinx.fem.assemble_vector(L)
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    c.value = [[1.0, 2.0], [3.0, 4.0]]

    A2 = dolfinx.fem.assemble_matrix(a)
    A2.assemble()

    b2 = dolfinx.fem.assemble_vector(L)
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    assert (A1 * 3.0 - A2 * 5.0).norm() == pytest.approx(0.0)
    assert (b1 * 3.0 - b2 * 5.0).norm() == pytest.approx(0.0)
