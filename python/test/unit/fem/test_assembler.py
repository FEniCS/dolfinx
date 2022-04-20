# Copyright (C) 2018-2019 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly"""

import math

import numpy as np
import pytest
import scipy.sparse

import ufl
from dolfinx import cpp as _cpp
from dolfinx import fem, graph, la
from dolfinx.fem import (Constant, Function, FunctionSpace,
                         VectorFunctionSpace, assemble_scalar, bcs_by_block,
                         dirichletbc, extract_function_spaces, form,
                         locate_dofs_geometrical, locate_dofs_topological)
from dolfinx.fem.petsc import (apply_lifting, apply_lifting_nest,
                               assemble_matrix, assemble_matrix_block,
                               assemble_matrix_nest, assemble_vector,
                               assemble_vector_block, assemble_vector_nest,
                               set_bc, set_bc_nest)
from dolfinx.mesh import (CellType, GhostMode, create_mesh, create_rectangle,
                          create_unit_cube, create_unit_square,
                          locate_entities_boundary)
from ufl import derivative, ds, dx, inner
from ufl.geometry import SpatialCoordinate

from mpi4py import MPI
from petsc4py import PETSc


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


@pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
def test_assemble_functional_dx(mode):
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, ghost_mode=mode)
    M = form(1.0 * dx(domain=mesh))
    value = assemble_scalar(M)
    value = mesh.comm.allreduce(value, op=MPI.SUM)
    assert value == pytest.approx(1.0, 1e-12)
    x = ufl.SpatialCoordinate(mesh)
    M = form(x[0] * dx(domain=mesh))
    value = assemble_scalar(M)
    value = mesh.comm.allreduce(value, op=MPI.SUM)
    assert value == pytest.approx(0.5, 1e-12)


@pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
def test_assemble_functional_ds(mode):
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, ghost_mode=mode)
    M = form(1.0 * ds(domain=mesh))
    value = assemble_scalar(M)
    value = mesh.comm.allreduce(value, op=MPI.SUM)
    assert value == pytest.approx(4.0, 1e-12)


def test_assemble_derivatives():
    """This test checks the original_coefficient_positions, which may change
    under differentiation (some coefficients and constants are
    eliminated)"""
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12)
    Q = FunctionSpace(mesh, ("Lagrange", 1))
    u = Function(Q)
    v = ufl.TestFunction(Q)
    du = ufl.TrialFunction(Q)
    b = Function(Q)
    c1 = Constant(mesh, np.array([[1.0, 0.0], [3.0, 4.0]], PETSc.ScalarType))
    c2 = Constant(mesh, PETSc.ScalarType(2.0))

    b.x.array[:] = 2.0

    # derivative eliminates 'u' and 'c1'
    L = ufl.inner(c1, c1) * v * dx + c2 * b * inner(u, v) * dx
    a = form(derivative(L, u, du))

    A1 = assemble_matrix(a)
    A1.assemble()
    a = form(c2 * b * inner(du, v) * dx)
    A2 = assemble_matrix(a)
    A2.assemble()
    assert (A1 - A2).norm() == pytest.approx(0.0, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
def test_basic_assembly(mode):
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, ghost_mode=mode)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    f = Function(V)
    f.x.array[:] = 10.0
    a = inner(f * u, v) * dx + inner(u, v) * ds
    L = inner(f, v) * dx + inner(2.0, v) * ds
    a, L = form(a), form(L)

    # Initial assembly
    A = assemble_matrix(a)
    A.assemble()
    assert isinstance(A, PETSc.Mat)
    b = assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert isinstance(b, PETSc.Vec)

    # Second assembly
    normA = A.norm()
    A.zeroEntries()
    A = assemble_matrix(A, a)
    A.assemble()
    assert isinstance(A, PETSc.Mat)
    assert normA == pytest.approx(A.norm())
    normb = b.norm()
    with b.localForm() as b_local:
        b_local.set(0.0)
    b = assemble_vector(b, L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert isinstance(b, PETSc.Vec)
    assert normb == pytest.approx(b.norm())

    # Vector re-assembly - no zeroing (but need to zero ghost entries)
    with b.localForm() as b_local:
        b_local.array[b.local_size:] = 0.0
    assemble_vector(b, L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert 2.0 * normb == pytest.approx(b.norm())

    # Matrix re-assembly (no zeroing)
    assemble_matrix(A, a)
    A.assemble()
    assert 2.0 * normA == pytest.approx(A.norm())


@pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
def test_basic_assembly_petsc_matrixcsr(mode):
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, ghost_mode=mode)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = inner(u, v) * dx + inner(u, v) * ds
    a = form(a)

    A0 = fem.assemble_matrix(a)
    A0.finalize()
    assert isinstance(A0, la.MatrixCSRMetaClass)
    A1 = fem.petsc.assemble_matrix(a)
    A1.assemble()
    assert isinstance(A1, PETSc.Mat)
    assert np.sqrt(A0.norm_squared()) == pytest.approx(A1.norm())

    V = VectorFunctionSpace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = form(inner(u, v) * dx + inner(u, v) * ds)
    with pytest.raises(RuntimeError):
        A0 = fem.assemble_matrix(a)


@pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
def test_assembly_bcs(mode):
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, ghost_mode=mode)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = inner(u, v) * dx + inner(u, v) * ds
    L = inner(1.0, v) * dx
    a, L = form(a), form(L)

    bdofsV = locate_dofs_geometrical(V, lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)))
    bc = dirichletbc(PETSc.ScalarType(1), bdofsV, V)

    # Assemble and apply 'global' lifting of bcs
    A = assemble_matrix(a)
    A.assemble()
    b = assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    g = b.duplicate()
    with g.localForm() as g_local:
        g_local.set(0.0)
    set_bc(g, [bc])
    f = b - A * g
    set_bc(f, [bc])

    # Assemble vector and apply lifting of bcs during assembly
    b_bc = assemble_vector(L)
    apply_lifting(b_bc, [a], [[bc]])
    b_bc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b_bc, [bc])

    assert (f - b_bc).norm() == pytest.approx(0.0, rel=1e-12, abs=1e-12)


@pytest.mark.skip_in_parallel
def test_assemble_manifold():
    """Test assembly of poisson problem on a mesh with topological
    dimension 1 but embedded in 2D (gdim=2)"""
    points = np.array([[0.0, 0.0], [0.2, 0.0], [0.4, 0.0],
                       [0.6, 0.0], [0.8, 0.0], [1.0, 0.0]], dtype=np.float64)
    cells = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.int32)
    cell = ufl.Cell("interval", geometric_dimension=points.shape[1])
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 1))
    mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
    assert mesh.geometry.dim == 2
    assert mesh.topology.dim == 1

    U = FunctionSpace(mesh, ("P", 1))
    u, v = ufl.TrialFunction(U), ufl.TestFunction(U)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx(mesh)
    L = ufl.inner(1.0, v) * ufl.dx(mesh)
    a, L = form(a), form(L)

    bcdofs = locate_dofs_geometrical(U, lambda x: np.isclose(x[0], 0.0))
    bcs = [dirichletbc(PETSc.ScalarType(0), bcdofs, U)]
    A = assemble_matrix(a, bcs=bcs)
    A.assemble()

    b = assemble_vector(L)
    apply_lifting(b, [a], bcs=[bcs])
    set_bc(b, bcs)

    assert np.isclose(b.norm(), 0.41231)
    assert np.isclose(A.norm(), 25.0199)


@ pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
def test_matrix_assembly_block(mode):
    """Test assembly of block matrices and vectors into (a) monolithic
    blocked structures, PETSc Nest structures, and monolithic
    structures"""
    mesh = create_unit_square(MPI.COMM_WORLD, 4, 8, ghost_mode=mode)
    p0, p1 = 1, 2
    P0 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p0)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p1)
    V0 = FunctionSpace(mesh, P0)
    V1 = FunctionSpace(mesh, P1)

    # Locate facets on boundary
    facetdim = mesh.topology.dim - 1
    bndry_facets = locate_entities_boundary(mesh, facetdim, lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                                                    np.isclose(x[0], 1.0)))
    bdofsV1 = locate_dofs_topological(V1, facetdim, bndry_facets)
    u_bc = PETSc.ScalarType(50.0)
    bc = dirichletbc(u_bc, bdofsV1, V1)

    # Define variational problem
    u, p = ufl.TrialFunction(V0), ufl.TrialFunction(V1)
    v, q = ufl.TestFunction(V0), ufl.TestFunction(V1)
    f = 1.0
    g = -3.0
    zero = Function(V0)

    a00 = inner(u, v) * dx
    a01 = inner(p, v) * dx
    a10 = inner(u, q) * dx
    a11 = inner(p, q) * dx

    L0 = zero * inner(f, v) * dx
    L1 = inner(g, q) * dx

    a_block = form([[a00, a01], [a10, a11]])
    L_block = form([L0, L1])

    # Monolithic blocked
    A0 = assemble_matrix_block(a_block, bcs=[bc])
    A0.assemble()
    b0 = assemble_vector_block(L_block, a_block, bcs=[bc])
    assert A0.getType() != "nest"
    Anorm0 = A0.norm()
    bnorm0 = b0.norm()

    # Nested (MatNest)
    A1 = assemble_matrix_nest(a_block, bcs=[bc], mat_types=[["baij", "aij"], ["aij", ""]])
    A1.assemble()
    Anorm1 = nest_matrix_norm(A1)
    assert Anorm0 == pytest.approx(Anorm1, 1.0e-12)

    b1 = assemble_vector_nest(L_block)
    apply_lifting_nest(b1, a_block, bcs=[bc])
    for b_sub in b1.getNestSubVecs():
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    bcs0 = bcs_by_block([L.function_spaces[0] for L in L_block], [bc])
    set_bc_nest(b1, bcs0)
    b1.assemble()

    bnorm1 = math.sqrt(sum([x.norm()**2 for x in b1.getNestSubVecs()]))
    assert bnorm0 == pytest.approx(bnorm1, 1.0e-12)

    # Monolithic version
    E = P0 * P1
    W = FunctionSpace(mesh, E)
    u0, u1 = ufl.TrialFunctions(W)
    v0, v1 = ufl.TestFunctions(W)
    a = inner(u0, v0) * dx + inner(u1, v1) * dx + inner(u0, v1) * dx + inner(
        u1, v0) * dx
    L = zero * inner(f, v0) * ufl.dx + inner(g, v1) * dx
    a, L = form(a), form(L)

    bdofsW_V1 = locate_dofs_topological(W.sub(1), mesh.topology.dim - 1, bndry_facets)
    bc = dirichletbc(u_bc, bdofsW_V1, W.sub(1))
    A2 = assemble_matrix(a, bcs=[bc])
    A2.assemble()
    b2 = assemble_vector(L)
    apply_lifting(b2, [a], bcs=[[bc]])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b2, [bc])
    assert A2.getType() != "nest"
    assert A2.norm() == pytest.approx(Anorm0, 1.0e-9)
    assert b2.norm() == pytest.approx(bnorm0, 1.0e-9)


@ pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
def test_assembly_solve_block(mode):
    """Solve a two-field mass-matrix like problem with block matrix approaches
    and test that solution is the same"""
    mesh = create_unit_square(MPI.COMM_WORLD, 32, 31, ghost_mode=mode)
    P = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    V0 = FunctionSpace(mesh, P)
    V1 = V0.clone()

    # Locate facets on boundary
    facetdim = mesh.topology.dim - 1
    bndry_facets = locate_entities_boundary(mesh, facetdim, lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                                                    np.isclose(x[0], 1.0)))

    bdofsV0 = locate_dofs_topological(V0, facetdim, bndry_facets)
    bdofsV1 = locate_dofs_topological(V1, facetdim, bndry_facets)

    u0_bc = PETSc.ScalarType(50.0)
    u1_bc = PETSc.ScalarType(20.0)
    bcs = [dirichletbc(u0_bc, bdofsV0, V0), dirichletbc(u1_bc, bdofsV1, V1)]

    # Variational problem
    u, p = ufl.TrialFunction(V0), ufl.TrialFunction(V1)
    v, q = ufl.TestFunction(V0), ufl.TestFunction(V1)
    f = 1.0
    g = -3.0
    zero = Function(V0)

    a00 = form(inner(u, v) * dx)
    a01 = form(zero * inner(p, v) * dx)
    a10 = form(zero * inner(u, q) * dx)
    a11 = form(inner(p, q) * dx)
    L0 = form(inner(f, v) * dx)
    L1 = form(inner(g, q) * dx)

    def monitor(ksp, its, rnorm):
        pass
        # print("Norm:", its, rnorm)

    A0 = assemble_matrix_block([[a00, a01], [a10, a11]], bcs=bcs)
    b0 = assemble_vector_block([L0, L1], [[a00, a01], [a10, a11]], bcs=bcs)
    A0.assemble()
    A0norm = A0.norm()
    b0norm = b0.norm()
    x0 = A0.createVecLeft()
    ksp = PETSc.KSP()
    ksp.create(mesh.comm)
    ksp.setOperators(A0)
    ksp.setMonitor(monitor)
    ksp.setType('cg')
    ksp.setTolerances(rtol=1.0e-14)
    ksp.setFromOptions()
    ksp.solve(b0, x0)
    x0norm = x0.norm()

    # Nested (MatNest)
    A1 = assemble_matrix_nest([[a00, a01], [a10, a11]], bcs=bcs, diagonal=1.0)
    A1.assemble()
    b1 = assemble_vector_nest([L0, L1])
    apply_lifting_nest(b1, [[a00, a01], [a10, a11]], bcs=bcs)
    for b_sub in b1.getNestSubVecs():
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    bcs0 = bcs_by_block([L0.function_spaces[0], L1.function_spaces[0]], bcs)
    set_bc_nest(b1, bcs0)
    b1.assemble()

    b1norm = b1.norm()
    assert b1norm == pytest.approx(b0norm, 1.0e-12)
    A1norm = nest_matrix_norm(A1)
    assert A0norm == pytest.approx(A1norm, 1.0e-12)

    x1 = b1.copy()
    ksp = PETSc.KSP()
    ksp.create(mesh.comm)
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
    W = FunctionSpace(mesh, E)
    u0, u1 = ufl.TrialFunctions(W)
    v0, v1 = ufl.TestFunctions(W)
    a = inner(u0, v0) * dx + inner(u1, v1) * dx
    L = inner(f, v0) * ufl.dx + inner(g, v1) * dx
    a, L = form(a), form(L)

    bdofsW0_V0 = locate_dofs_topological(W.sub(0), facetdim, bndry_facets)
    bdofsW1_V1 = locate_dofs_topological(W.sub(1), facetdim, bndry_facets)
    bcs = [dirichletbc(u0_bc, bdofsW0_V0, W.sub(0)), dirichletbc(u1_bc, bdofsW1_V1, W.sub(1))]

    A2 = assemble_matrix(a, bcs=bcs)
    A2.assemble()
    b2 = assemble_vector(L)
    apply_lifting(b2, [a], [bcs])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b2, bcs)
    A2norm = A2.norm()
    b2norm = b2.norm()
    assert A2norm == pytest.approx(A0norm, 1.0e-12)
    assert b2norm == pytest.approx(b0norm, 1.0e-12)

    x2 = b2.copy()
    ksp = PETSc.KSP()
    ksp.create(mesh.comm)
    ksp.setMonitor(monitor)
    ksp.setOperators(A2)
    ksp.setType('cg')
    ksp.getPC().setType('jacobi')
    ksp.setTolerances(rtol=1.0e-12)
    ksp.setFromOptions()
    ksp.solve(b2, x2)
    x2norm = x2.norm()
    assert x2norm == pytest.approx(x0norm, 1.0e-10)


@ pytest.mark.parametrize("mesh", [
    create_unit_square(MPI.COMM_WORLD, 12, 11, ghost_mode=GhostMode.none),
    create_unit_square(MPI.COMM_WORLD, 12, 11, ghost_mode=GhostMode.shared_facet),
    create_unit_cube(MPI.COMM_WORLD, 3, 7, 3, ghost_mode=GhostMode.none),
    create_unit_cube(MPI.COMM_WORLD, 3, 7, 3, ghost_mode=GhostMode.shared_facet)
])
def test_assembly_solve_taylor_hood(mesh):
    """Assemble Stokes problem with Taylor-Hood elements and solve."""
    P2 = VectorFunctionSpace(mesh, ("Lagrange", 2))
    P1 = FunctionSpace(mesh, ("Lagrange", 1))

    def boundary0(x):
        """Define boundary x = 0"""
        return np.isclose(x[0], 0.0)

    def boundary1(x):
        """Define boundary x = 1"""
        return np.isclose(x[0], 1.0)

    # Locate facets on boundaries
    facetdim = mesh.topology.dim - 1
    bndry_facets0 = locate_entities_boundary(mesh, facetdim, boundary0)
    bndry_facets1 = locate_entities_boundary(mesh, facetdim, boundary1)

    bdofs0 = locate_dofs_topological(P2, facetdim, bndry_facets0)
    bdofs1 = locate_dofs_topological(P2, facetdim, bndry_facets1)

    bc_value = np.ones(mesh.geometry.dim, dtype=PETSc.ScalarType)
    bc0 = dirichletbc(bc_value, bdofs0, P2)
    bc1 = dirichletbc(bc_value, bdofs1, P2)

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
    p_zero = Function(P1)
    f = Function(P2)
    L0 = ufl.inner(f, v) * dx
    L1 = ufl.inner(p_zero, q) * dx

    def nested_solve():
        """Nested solver"""
        A = assemble_matrix_nest(form([[a00, a01], [a10, a11]]), bcs=[bc0, bc1],
                                 mat_types=[["baij", "aij"], ["aij", ""]])
        A.assemble()
        P = assemble_matrix_nest(form([[p00, p01], [p10, p11]]), bcs=[bc0, bc1],
                                 mat_types=[["aij", "aij"], ["aij", ""]])
        P.assemble()
        b = assemble_vector_nest(form([L0, L1]))
        apply_lifting_nest(b, form([[a00, a01], [a10, a11]]), [bc0, bc1])
        for b_sub in b.getNestSubVecs():
            b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        bcs = bcs_by_block(extract_function_spaces(form([L0, L1])), [bc0, bc1])
        set_bc_nest(b, bcs)
        b.assemble()

        ksp = PETSc.KSP()
        ksp.create(mesh.comm)
        ksp.setOperators(A, P)
        nested_IS = P.getNestISs()
        ksp.setType("minres")
        pc = ksp.getPC()
        pc.setType("fieldsplit")
        pc.setFieldSplitIS(["u", nested_IS[0][0]], ["p", nested_IS[1][1]])
        ksp_u, ksp_p = pc.getFieldSplitSubKSP()
        ksp_u.setType("preonly")
        ksp_u.getPC().setType('lu')
        ksp_p.setType("preonly")

        def monitor(ksp, its, rnorm):
            # print("Num it, rnorm:", its, rnorm)
            pass

        ksp.setTolerances(rtol=1.0e-8, max_it=50)
        ksp.setMonitor(monitor)
        ksp.setFromOptions()
        x = b.copy()
        ksp.solve(b, x)
        assert ksp.getConvergedReason() > 0
        return b.norm(), x.norm(), nest_matrix_norm(A), nest_matrix_norm(P)

    def blocked_solve():
        """Blocked (monolithic) solver"""
        A = assemble_matrix_block(form([[a00, a01], [a10, a11]]), bcs=[bc0, bc1])
        A.assemble()
        P = assemble_matrix_block(form([[p00, p01], [p10, p11]]), bcs=[bc0, bc1])
        P.assemble()
        b = assemble_vector_block(form([L0, L1]), form([[a00, a01], [a10, a11]]), bcs=[bc0, bc1])

        ksp = PETSc.KSP()
        ksp.create(mesh.comm)
        ksp.setOperators(A, P)
        ksp.setType("minres")
        pc = ksp.getPC()
        pc.setType('lu')
        ksp.setTolerances(rtol=1.0e-8, max_it=50)
        ksp.setFromOptions()
        x = A.createVecRight()
        ksp.solve(b, x)
        assert ksp.getConvergedReason() > 0
        return b.norm(), x.norm(), A.norm(), P.norm()

    def monolithic_solve():
        """Monolithic (interleaved) solver"""
        P2_el = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
        P1_el = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        TH = P2_el * P1_el
        W = FunctionSpace(mesh, TH)
        (u, p) = ufl.TrialFunctions(W)
        (v, q) = ufl.TestFunctions(W)
        a00 = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
        a01 = ufl.inner(p, ufl.div(v)) * dx
        a10 = ufl.inner(ufl.div(u), q) * dx
        a = a00 + a01 + a10

        p00 = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
        p11 = ufl.inner(p, q) * dx
        p_form = p00 + p11

        f = Function(W.sub(0).collapse()[0])
        p_zero = Function(W.sub(1).collapse()[0])
        L0 = inner(f, v) * dx
        L1 = inner(p_zero, q) * dx
        L = L0 + L1

        a, p_form, L = form(a), form(p_form), form(L)

        bdofsW0_P2_0 = locate_dofs_topological((W.sub(0), P2), facetdim, bndry_facets0)
        bdofsW0_P2_1 = locate_dofs_topological((W.sub(0), P2), facetdim, bndry_facets1)
        u0 = Function(P2)
        u0.x.array[:] = 1.0
        bc0 = dirichletbc(u0, bdofsW0_P2_0, W.sub(0))
        bc1 = dirichletbc(u0, bdofsW0_P2_1, W.sub(0))

        A = assemble_matrix(a, bcs=[bc0, bc1])
        A.assemble()
        P = assemble_matrix(p_form, bcs=[bc0, bc1])
        P.assemble()

        b = assemble_vector(L)
        apply_lifting(b, [a], bcs=[[bc0, bc1]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [bc0, bc1])

        ksp = PETSc.KSP()
        ksp.create(mesh.comm)
        ksp.setOperators(A, P)
        ksp.setType("minres")
        pc = ksp.getPC()
        pc.setType('lu')

        def monitor(ksp, its, rnorm):
            # print("Num it, rnorm:", its, rnorm)
            pass

        ksp.setTolerances(rtol=1.0e-8, max_it=50)
        ksp.setMonitor(monitor)
        ksp.setFromOptions()
        x = A.createVecRight()
        ksp.solve(b, x)
        assert ksp.getConvergedReason() > 0
        return b.norm(), x.norm(), A.norm(), P.norm()

    bnorm0, xnorm0, Anorm0, Pnorm0 = nested_solve()
    bnorm1, xnorm1, Anorm1, Pnorm1 = blocked_solve()
    bnorm2, xnorm2, Anorm2, Pnorm2 = monolithic_solve()

    assert bnorm1 == pytest.approx(bnorm0, 1.0e-12)
    assert xnorm1 == pytest.approx(xnorm0, 1.0e-8)
    assert Anorm1 == pytest.approx(Anorm0, 1.0e-12)
    assert Pnorm1 == pytest.approx(Pnorm0, 1.0e-12)

    assert bnorm2 == pytest.approx(bnorm0, 1.0e-12)
    assert xnorm2 == pytest.approx(xnorm0, 1.0e-8)
    assert Anorm2 == pytest.approx(Anorm0, 1.0e-12)
    assert Pnorm2 == pytest.approx(Pnorm0, 1.0e-12)


def test_basic_interior_facet_assembly():
    mesh = create_rectangle(MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
                            [5, 5], cell_type=CellType.triangle,
                            ghost_mode=GhostMode.shared_facet)
    V = FunctionSpace(mesh, ("DG", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(ufl.avg(u), ufl.avg(v)) * ufl.dS
    a = form(a)
    A = assemble_matrix(a)
    A.assemble()
    assert isinstance(A, PETSc.Mat)

    L = ufl.conj(ufl.avg(v)) * ufl.dS
    L = form(L)
    b = assemble_vector(L)
    b.assemble()
    assert isinstance(b, PETSc.Vec)


@ pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
def test_basic_assembly_constant(mode):
    """Tests assembly with Constant

    The following test should be sensitive to order of flattening the
    matrix-valued constant.

    """
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5, ghost_mode=mode)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    c = Constant(mesh, np.array([[1.0, 2.0], [5.0, 3.0]], PETSc.ScalarType))

    a = inner(c[1, 0] * u, v) * dx + inner(c[1, 0] * u, v) * ds
    L = inner(c[1, 0], v) * dx + inner(c[1, 0], v) * ds
    a, L = form(a), form(L)

    # Initial assembly
    A1 = assemble_matrix(a)
    A1.assemble()

    b1 = assemble_vector(L)
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    c.value = [[1.0, 2.0], [3.0, 4.0]]

    A2 = assemble_matrix(a)
    A2.assemble()

    b2 = assemble_vector(L)
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    assert (A1 * 3.0 - A2 * 5.0).norm() == pytest.approx(0.0)
    assert (b1 * 3.0 - b2 * 5.0).norm() == pytest.approx(0.0)


def test_lambda_assembler():
    """Tests assembly with a lambda function"""
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    a = inner(u, v) * dx

    # Initial assembly
    a_form = form(a)

    rdata = []
    cdata = []
    vdata = []

    def mat_insert(rows, cols, vals):
        vdata.append(vals)
        rdata.append(np.repeat(rows, len(cols)))
        cdata.append(np.tile(cols, len(rows)))
        return 0

    _cpp.fem.assemble_matrix(mat_insert, a_form, [])
    vdata = np.array(vdata).flatten()
    cdata = np.array(cdata).flatten()
    rdata = np.array(rdata).flatten()
    mat = scipy.sparse.coo_matrix((vdata, (rdata, cdata)))
    v = np.ones(mat.shape[1])
    s = MPI.COMM_WORLD.allreduce(mat.dot(v).sum(), MPI.SUM)
    assert np.isclose(s, 1.0)


def test_pack_coefficients():
    """Test packing of form coefficients ahead of main assembly call"""
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 15)
    V = FunctionSpace(mesh, ("Lagrange", 1))

    # Non-blocked
    u = Function(V)
    v = ufl.TestFunction(V)
    c = Constant(mesh, PETSc.ScalarType(12.0))
    F = ufl.inner(c, v) * dx - c * ufl.sqrt(u * u) * ufl.inner(u, v) * dx
    u.x.array[:] = 10.0
    _F = form(F)

    # -- Test vector
    b0 = assemble_vector(_F)
    b0.assemble()
    constants = _cpp.fem.pack_constants(_F)
    coeffs = _cpp.fem.pack_coefficients(_F)
    with b0.localForm() as _b0:
        for c in [(None, None), (None, coeffs), (constants, None), (constants, coeffs)]:
            b = assemble_vector(_F, c[0], c[1])
            b.assemble()
            with b.localForm() as _b:
                assert (_b0.array_r == _b.array_r).all()

    # Change coefficients
    constants *= 5.0
    for coeff in coeffs.values():
        coeff *= 5.0
    with b0.localForm() as _b0:
        for c in [(None, coeffs), (constants, None), (constants, coeffs)]:
            b = assemble_vector(_F, c[0], c[1])
            b.assemble()
            with b.localForm() as _b:
                assert (_b0 - _b).norm() > 1.0e-5

    # -- Test matrix
    du = ufl.TrialFunction(V)
    J = ufl.derivative(F, u, du)
    J = form(J)

    A0 = assemble_matrix(J)
    A0.assemble()

    constants = _cpp.fem.pack_constants(J)
    coeffs = _cpp.fem.pack_coefficients(J)
    for c in [(None, None), (None, coeffs), (constants, None), (constants, coeffs)]:
        A = assemble_matrix(J, constants=c[0], coeffs=c[1])
        A.assemble()
        assert pytest.approx((A - A0).norm(), 1.0e-12) == 0.0

    # Change coefficients
    constants *= 5.0
    for coeff in coeffs.values():
        coeff *= 5.0
    for c in [(None, coeffs), (constants, None), (constants, coeffs)]:
        A = assemble_matrix(J, constants=c[0], coeffs=c[1])
        A.assemble()
        assert (A - A0).norm() > 1.0e-5


def test_coefficents_non_constant():
    "Test packing coefficients with non-constant values"
    mesh = create_unit_square(MPI.COMM_WORLD, 3, 5)
    V = FunctionSpace(mesh, ("Lagrange", 3))  # degree 3 so that interpolation is exact

    u = Function(V)
    u.interpolate(lambda x: x[0] * x[1]**2)
    x = SpatialCoordinate(mesh)

    v = ufl.TestFunction(V)

    # -- Volume integral vector
    F = form((ufl.inner(u, v) - ufl.inner(x[0] * x[1]**2, v)) * dx)
    b0 = assemble_vector(F)
    b0.assemble()
    assert np.linalg.norm(b0.array) == pytest.approx(0.0)

    # -- Exterior facet integral vector
    F = form((ufl.inner(u, v) - ufl.inner(x[0] * x[1]**2, v)) * ds)
    b0 = assemble_vector(F)
    b0.assemble()
    assert np.linalg.norm(b0.array) == pytest.approx(0.0)

    # -- Interior facet integral vector
    V = FunctionSpace(mesh, ("DG", 3))  # degree 3 so that interpolation is exact

    u0 = Function(V)
    u0.interpolate(lambda x: x[1]**2)
    u1 = Function(V)
    u1.interpolate(lambda x: x[0])
    x = SpatialCoordinate(mesh)

    v = ufl.TestFunction(V)

    F = (ufl.inner(u1('+') * u0('-'), ufl.avg(v)) - ufl.inner(x[0] * x[1]**2, ufl.avg(v))) * ufl.dS
    F = form(F)
    b0 = assemble_vector(F)
    b0.assemble()
    assert np.linalg.norm(b0.array) == pytest.approx(0.0)


def test_vector_types():
    """Assemble form using different types"""
    mesh = create_unit_square(MPI.COMM_WORLD, 3, 5)
    V = FunctionSpace(mesh, ("Lagrange", 3))
    v = ufl.TestFunction(V)

    c = Constant(mesh, np.float64(1))
    L = inner(c, v) * ufl.dx
    x0 = la.vector(V.dofmap.index_map, V.dofmap.index_map_bs, dtype=np.float64)
    L = form(L, dtype=x0.array.dtype)
    c0 = _cpp.fem.pack_constants(L)
    c1 = _cpp.fem.pack_coefficients(L)
    _cpp.fem.assemble_vector(x0.array, L, c0, c1)
    x0.scatter_reverse(_cpp.common.ScatterMode.add)

    c = Constant(mesh, np.complex128(1))
    L = inner(c, v) * ufl.dx
    x1 = la.vector(V.dofmap.index_map, V.dofmap.index_map_bs, dtype=np.complex128)
    L = form(L, dtype=x1.array.dtype)
    c0 = _cpp.fem.pack_constants(L)
    c1 = _cpp.fem.pack_coefficients(L)
    _cpp.fem.assemble_vector(x1.array, L, c0, c1)
    x1.scatter_reverse(_cpp.common.ScatterMode.add)

    c = Constant(mesh, np.float32(1))
    L = inner(c, v) * ufl.dx
    x2 = la.vector(V.dofmap.index_map, V.dofmap.index_map_bs, dtype=np.float32)
    L = form(L, dtype=x2.array.dtype)
    c0 = _cpp.fem.pack_constants(L)
    c1 = _cpp.fem.pack_coefficients(L)
    _cpp.fem.assemble_vector(x2.array, L, c0, c1)
    x2.scatter_reverse(_cpp.common.ScatterMode.add)

    assert np.linalg.norm(x0.array - x1.array) == pytest.approx(0.0)
    assert np.linalg.norm(x0.array - x2.array) == pytest.approx(0.0, abs=1e-8)


def test_assemble_empty_rank_mesh():
    """Assembly on mesh where some ranks are empty"""
    comm = MPI.COMM_WORLD
    cell_type = CellType.triangle
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", ufl.Cell(cell_type.name), 1))

    def partitioner(comm, nparts, local_graph, num_ghost_nodes, ghosting):
        """Leave cells on the curent rank"""
        dest = np.full(len(cells), comm.rank, dtype=np.int32)
        return graph.create_adjacencylist(dest)

    if comm.rank == 0:
        # Put cells on rank 0
        cells = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        cells = graph.create_adjacencylist(cells)
        x = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    else:
        # No cells onm other ranks
        cells = graph.create_adjacencylist(np.empty((0, 3), dtype=np.int64))
        x = np.empty((0, 2), dtype=np.float64)

    mesh = create_mesh(comm, cells, x, domain, GhostMode.none, partitioner)

    V = FunctionSpace(mesh, ("Lagrange", 2))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    f, k, zero = Function(V), Function(V), Function(V)
    f.x.array[:] = 10.0
    k.x.array[:] = 1.0
    zero.x.array[:] = 0.0
    a = form(inner(k * u, v) * dx + inner(zero * u, v) * ds)
    L = form(inner(f, v) * dx + inner(zero, v) * ds)
    M = form(2 * k * dx + k * ds)

    sum = comm.allreduce(assemble_scalar(M), op=MPI.SUM)
    assert sum == pytest.approx(6.0)

    # Assemble
    A = assemble_matrix(a)
    A.assemble()
    b = assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Solve
    ksp = PETSc.KSP()
    ksp.create(mesh.comm)
    ksp.setOperators(A)
    ksp.setTolerances(rtol=1.0e-9, max_it=50)
    ksp.setFromOptions()
    x = b.copy()
    ksp.solve(b, x)

    assert np.allclose(x.array, 10.0)
