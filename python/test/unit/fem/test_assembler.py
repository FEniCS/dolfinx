# Copyright (C) 2018-2022 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly"""

import math

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import pytest
import scipy.sparse

import basix
import ufl
from basix.ufl import element, mixed_element
from dolfinx import cpp as _cpp
from dolfinx import default_real_type, fem, graph, la
from dolfinx.fem import (
    Constant,
    Function,
    assemble_scalar,
    bcs_by_block,
    dirichletbc,
    extract_function_spaces,
    form,
    functionspace,
    locate_dofs_geometrical,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import apply_lifting as petsc_apply_lifting
from dolfinx.fem.petsc import apply_lifting_nest as petsc_apply_lifting_nest
from dolfinx.fem.petsc import assemble_matrix as petsc_assemble_matrix
from dolfinx.fem.petsc import assemble_matrix_block as petsc_assemble_matrix_block
from dolfinx.fem.petsc import assemble_matrix_nest as petsc_assemble_matrix_nest
from dolfinx.fem.petsc import assemble_vector as petsc_assemble_vector
from dolfinx.fem.petsc import assemble_vector_block as petsc_assemble_vector_block
from dolfinx.fem.petsc import assemble_vector_nest as petsc_assemble_vector_nest
from dolfinx.fem.petsc import set_bc as petsc_set_bc
from dolfinx.fem.petsc import set_bc_nest as petsc_set_bc_nest
from dolfinx.mesh import (
    CellType,
    GhostMode,
    create_mesh,
    create_rectangle,
    create_unit_cube,
    create_unit_square,
    locate_entities_boundary,
)
from ufl import derivative, dS, ds, dx, inner
from ufl.geometry import SpatialCoordinate


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
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_assemble_functional_dx(mode, dtype):
    xtype = dtype(0).real.dtype
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, ghost_mode=mode, dtype=xtype)
    M = form(1.0 * dx(domain=mesh), dtype=dtype)
    value = assemble_scalar(M)
    value = mesh.comm.allreduce(value, op=MPI.SUM)
    assert value == pytest.approx(1.0, 1e-5)
    x = ufl.SpatialCoordinate(mesh)
    M = form(x[0] * dx(domain=mesh), dtype=dtype)
    value = assemble_scalar(M)
    value = mesh.comm.allreduce(value, op=MPI.SUM)
    assert value == pytest.approx(0.5, 1e-6)


@pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_assemble_functional_ds(mode, dtype):
    xtype = dtype(0).real.dtype
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, ghost_mode=mode, dtype=xtype)
    M = form(1.0 * ds(domain=mesh), dtype=dtype)
    value = assemble_scalar(M)
    value = mesh.comm.allreduce(value, op=MPI.SUM)
    assert value == pytest.approx(4.0, 1e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_assemble_derivatives(dtype):
    """This test checks the original_coefficient_positions, which may change
    under differentiation (some coefficients and constants are
    eliminated)"""
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, dtype=dtype(0).real.dtype)
    Q = functionspace(mesh, ("Lagrange", 1))
    u = Function(Q, dtype=dtype)
    v = ufl.TestFunction(Q)
    du = ufl.TrialFunction(Q)
    b = Function(Q, dtype=dtype)
    c1 = Constant(mesh, np.array([[1.0, 0.0], [3.0, 4.0]], dtype=dtype))
    c2 = Constant(mesh, dtype(2.0))

    b.x.array[:] = 2.0

    # derivative eliminates 'u' and 'c1'
    L = ufl.inner(c1, c1) * v * dx + c2 * b * inner(u, v) * dx
    a = form(derivative(L, u, du), dtype=dtype)

    A1 = fem.assemble_matrix(a)
    A1.scatter_reverse()
    a = form(c2 * b * inner(du, v) * dx, dtype=dtype)
    A2 = fem.assemble_matrix(a)
    A2.scatter_reverse()
    assert np.allclose(A1.data, A2.data)


@pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_basic_assembly(mode, dtype):
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, ghost_mode=mode, dtype=dtype(0).real.dtype)
    V = functionspace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    f = Function(V, dtype=dtype)
    f.x.array[:] = 10.0
    a = inner(f * u, v) * dx + inner(u, v) * ds
    L = inner(f, v) * dx + inner(2.0, v) * ds
    a, L = form(a, dtype=dtype), form(L, dtype=dtype)

    # Initial assembly
    A = fem.assemble_matrix(a)
    A.scatter_reverse()
    assert isinstance(A, la.MatrixCSR)
    b = fem.assemble_vector(L)
    b.scatter_reverse(la.InsertMode.add)
    assert isinstance(b, la.Vector)

    # Second assembly
    normA = A.squared_norm()
    A.set_value(0)
    A = fem.assemble_matrix(A, a)
    A.scatter_reverse()
    assert isinstance(A, la.MatrixCSR)
    assert normA == pytest.approx(A.squared_norm())
    normb = la.norm(b)
    b.array[:] = 0
    fem.assemble_vector(b.array, L)
    b.scatter_reverse(la.InsertMode.add)
    assert normb == pytest.approx(la.norm(b))

    # Vector re-assembly - no zeroing (but need to zero ghost entries)
    b.array[b.index_map.size_local * b.block_size :] = 0
    fem.assemble_vector(b.array, L)
    b.scatter_reverse(la.InsertMode.add)
    assert 2 * normb == pytest.approx(la.norm(b))

    # Matrix re-assembly (no zeroing)
    fem.assemble_matrix(A, a)
    A.scatter_reverse()
    assert 4 * normA == pytest.approx(A.squared_norm())


@pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
def test_basic_assembly_petsc_matrixcsr(mode):
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, ghost_mode=mode)
    V = functionspace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = form(inner(u, v) * dx + inner(u, v) * ds)

    A0 = fem.assemble_matrix(a)
    A0.scatter_reverse()
    assert isinstance(A0, la.MatrixCSR)
    A1 = petsc_assemble_matrix(a)
    A1.assemble()
    assert isinstance(A1, PETSc.Mat)
    assert np.sqrt(A0.squared_norm()) == pytest.approx(A1.norm(), 1.0e-5)
    A1.destroy()

    gdim = mesh.geometry.dim
    V = functionspace(mesh, ("Lagrange", 1, (gdim,)))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = form(inner(u, v) * dx + inner(u, v) * ds)
    A0 = fem.assemble_matrix(a)
    A0.scatter_reverse()
    assert isinstance(A0, la.MatrixCSR)
    A1 = petsc_assemble_matrix(a)
    A1.assemble()
    assert isinstance(A1, PETSc.Mat)
    assert np.sqrt(A0.squared_norm()) == pytest.approx(A1.norm(), rel=1.0e-8, abs=1.0e-5)
    A1.destroy()


@pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
def test_assembly_bcs(mode):
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, ghost_mode=mode)
    V = functionspace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = form(inner(u, v) * dx + inner(u, v) * ds)
    L = form(inner(1.0, v) * dx)

    bdofsV = locate_dofs_geometrical(
        V, lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))
    )
    bc = dirichletbc(PETSc.ScalarType(1), bdofsV, V)

    # Assemble and apply 'global' lifting of bcs
    A = petsc_assemble_matrix(a)
    A.assemble()
    b = petsc_assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    g = b.duplicate()
    with g.localForm() as g_local:
        g_local.set(0.0)
    petsc_set_bc(g, [bc])
    # f = b - A * g
    f = b.duplicate()
    A.multAdd(-g, b, f)
    petsc_set_bc(f, [bc])

    # Assemble vector and apply lifting of bcs during assembly
    b_bc = petsc_assemble_vector(L)
    petsc_apply_lifting(b_bc, [a], [[bc]])
    b_bc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc_set_bc(b_bc, [bc])
    assert (f - b_bc).norm() == pytest.approx(0.0, rel=1e-6, abs=1e-6)
    A.destroy(), b.destroy(), g.destroy()


@pytest.mark.skip_in_parallel
def test_petsc_assemble_manifold():
    """Test assembly of poisson problem on a mesh with topological
    dimension 1 but embedded in 2D (gdim=2)"""
    points = np.array(
        [[0.0, 0.0], [0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0], [1.0, 0.0]],
        dtype=default_real_type,
    )
    cells = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.int32)
    domain = ufl.Mesh(
        element(
            basix.ElementFamily.P,
            basix.CellType.interval,
            1,
            shape=(points.shape[1],),
            dtype=default_real_type,
        )
    )
    mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
    assert mesh.geometry.dim == 2
    assert mesh.topology.dim == 1

    U = functionspace(mesh, ("P", 1))
    u, v = ufl.TrialFunction(U), ufl.TestFunction(U)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx(mesh)
    L = ufl.inner(1.0, v) * ufl.dx(mesh)
    a = form(a)
    L = form(L)

    bcdofs = locate_dofs_geometrical(U, lambda x: np.isclose(x[0], 0.0))
    bcs = [dirichletbc(PETSc.ScalarType(0), bcdofs, U)]
    A = petsc_assemble_matrix(a, bcs=bcs)
    A.assemble()

    b = petsc_assemble_vector(L)
    petsc_apply_lifting(b, [a], bcs=[bcs])
    petsc_set_bc(b, bcs)

    assert np.isclose(b.norm(), 0.41231)
    assert np.isclose(A.norm(), 25.0199)
    A.destroy(), b.destroy()


@pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
def test_matrix_assembly_block(mode):
    """Test assembly of block matrices and vectors into (a) monolithic
    blocked structures, PETSc Nest structures, and monolithic
    structures"""
    mesh = create_unit_square(MPI.COMM_WORLD, 4, 8, ghost_mode=mode)
    p0, p1 = 1, 2
    P0 = element("Lagrange", mesh.basix_cell(), p0)
    P1 = element("Lagrange", mesh.basix_cell(), p1)
    P2 = element("Lagrange", mesh.basix_cell(), p0)
    V0 = functionspace(mesh, P0)
    V1 = functionspace(mesh, P1)
    V2 = functionspace(mesh, P2)

    # Locate facets on boundary
    facetdim = mesh.topology.dim - 1
    bndry_facets = locate_entities_boundary(
        mesh, facetdim, lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))
    )
    bdofsV1 = locate_dofs_topological(V1, facetdim, bndry_facets)
    u_bc = PETSc.ScalarType(50.0)
    bc = dirichletbc(u_bc, bdofsV1, V1)

    # Define variational problem
    u, p, r = ufl.TrialFunction(V0), ufl.TrialFunction(V1), ufl.TrialFunction(V2)
    v, q, s = ufl.TestFunction(V0), ufl.TestFunction(V1), ufl.TestFunction(V2)
    f = 1.0
    g = -3.0
    zero = Function(V0)

    a00 = inner(u, v) * dx
    a01 = inner(p, v) * dx
    a02 = inner(r, v) * dx
    a10 = inner(u, q) * dx
    a11 = inner(p, q) * dx
    a12 = inner(r, q) * dx
    a20 = inner(u, s) * dx
    a21 = inner(p, s) * dx
    a22 = inner(r, s) * dx

    L0 = zero * inner(f, v) * dx
    L1 = inner(g, q) * dx
    L2 = inner(g, s) * dx

    a_block = form([[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]])
    L_block = form([L0, L1, L2])

    # Prepare a block problem with "None" on (1, 1) diagonal
    a_block_none = form([[a00, a01, a02], [None, None, a12], [a20, a21, a22]])

    def blocked():
        """Monolithic blocked"""
        A = petsc_assemble_matrix_block(a_block, bcs=[bc])
        A.assemble()
        b = petsc_assemble_vector_block(L_block, a_block, bcs=[bc])
        assert A.getType() != "nest"
        Anorm = A.norm()
        bnorm = b.norm()
        A.destroy(), b.destroy()
        with pytest.raises(RuntimeError):
            petsc_assemble_matrix_block(a_block_none, bcs=[bc])
        return Anorm, bnorm

    def nest():
        """Nested (MatNest)"""
        A = petsc_assemble_matrix_nest(
            a_block,
            bcs=[bc],
            mat_types=[["baij", "aij", "aij"], ["aij", "", "aij"], ["aij", "aij", "aij"]],
        )
        A.assemble()
        with pytest.raises(RuntimeError):
            petsc_assemble_matrix_nest(a_block_none, bcs=[bc])

        b = petsc_assemble_vector_nest(L_block)
        petsc_apply_lifting_nest(b, a_block, bcs=[bc])
        for b_sub in b.getNestSubVecs():
            b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        bcs0 = bcs_by_block([L.function_spaces[0] for L in L_block], [bc])
        petsc_set_bc_nest(b, bcs0)
        b.assemble()
        bnorm = math.sqrt(sum([x.norm() ** 2 for x in b.getNestSubVecs()]))
        Anorm = nest_matrix_norm(A)
        A.destroy(), b.destroy()
        return Anorm, bnorm

    def monolithic():
        """Monolithic version"""
        W = functionspace(mesh, mixed_element([P0, P1, P2]))
        u0, u1, u2 = ufl.TrialFunctions(W)
        v0, v1, v2 = ufl.TestFunctions(W)
        a = (
            inner(u0, v0) * dx
            + inner(u1, v1) * dx
            + inner(u0, v1) * dx
            + inner(u1, v0) * dx
            + inner(u0, v2) * dx
            + inner(u1, v2) * dx
            + inner(u2, v2) * dx
            + inner(u2, v0) * dx
            + inner(u2, v1) * dx
        )
        L = zero * inner(f, v0) * ufl.dx + inner(g, v1) * dx + inner(g, v2) * dx
        a, L = form(a), form(L)

        bdofsW_V1 = locate_dofs_topological(W.sub(1), mesh.topology.dim - 1, bndry_facets)
        bc = dirichletbc(u_bc, bdofsW_V1, W.sub(1))
        A = petsc_assemble_matrix(a, bcs=[bc])
        A.assemble()
        b = petsc_assemble_vector(L)
        petsc_apply_lifting(b, [a], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc_set_bc(b, [bc])
        assert A.getType() != "nest"
        Anorm = A.norm()
        bnorm = b.norm()
        A.destroy(), b.destroy()
        return Anorm, bnorm

    Anorm0, bnorm0 = blocked()
    Anorm1, bnorm1 = nest()
    assert Anorm1 == pytest.approx(Anorm0, 1.0e-4)
    assert bnorm1 == pytest.approx(bnorm0, 1.0e-6)

    Anorm2, bnorm2 = monolithic()
    assert Anorm2 == pytest.approx(Anorm0, 1.0e-4)
    assert bnorm2 == pytest.approx(bnorm0, 1.0e-6)


@pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
def test_assembly_solve_block(mode):
    """Solve a two-field mass-matrix like problem with block matrix approaches
    and test that solution is the same"""
    mesh = create_unit_square(MPI.COMM_WORLD, 32, 31, ghost_mode=mode)
    P = element("Lagrange", mesh.basix_cell(), 1)
    V0 = functionspace(mesh, P)
    V1 = V0.clone()

    # Locate facets on boundary
    facetdim = mesh.topology.dim - 1
    bndry_facets = locate_entities_boundary(
        mesh, facetdim, lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))
    )

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

    def blocked():
        """Blocked"""
        A = petsc_assemble_matrix_block([[a00, a01], [a10, a11]], bcs=bcs)
        b = petsc_assemble_vector_block([L0, L1], [[a00, a01], [a10, a11]], bcs=bcs)
        A.assemble()
        x = A.createVecLeft()
        ksp = PETSc.KSP()
        ksp.create(mesh.comm)
        ksp.setOperators(A)
        ksp.setMonitor(monitor)
        ksp.setType("cg")
        ksp.setTolerances(rtol=1.0e-14)
        ksp.setFromOptions()
        ksp.solve(b, x)

        Anorm = A.norm()
        bnorm = b.norm()
        xnorm = x.norm()
        ksp.destroy(), A.destroy(), b.destroy(), x.destroy()
        return Anorm, bnorm, xnorm

    def nested():
        """Nested (MatNest)"""
        A = petsc_assemble_matrix_nest([[a00, a01], [a10, a11]], bcs=bcs, diagonal=1.0)
        A.assemble()
        b = petsc_assemble_vector_nest([L0, L1])
        petsc_apply_lifting_nest(b, [[a00, a01], [a10, a11]], bcs=bcs)
        for b_sub in b.getNestSubVecs():
            b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        bcs0 = bcs_by_block([L0.function_spaces[0], L1.function_spaces[0]], bcs)
        petsc_set_bc_nest(b, bcs0)
        b.assemble()

        x = b.copy()
        ksp = PETSc.KSP()
        ksp.create(mesh.comm)
        ksp.setMonitor(monitor)
        ksp.setOperators(A)
        ksp.setType("cg")
        ksp.setTolerances(rtol=1.0e-12)
        ksp.setFromOptions()
        ksp.solve(b, x)

        Anorm = nest_matrix_norm(A)
        # bnorm = b.norm()
        bnorm = 0.0
        for b_sub in b.getNestSubVecs():
            bnorm += b_sub.norm() ** 2
        bnorm = np.sqrt(bnorm)
        xnorm = x.norm()
        ksp.destroy(), A.destroy(), b.destroy(), x.destroy()

        return Anorm, bnorm, xnorm

    def monolithic():
        """Monolithic version"""
        E = mixed_element([P, P])
        W = functionspace(mesh, E)
        u0, u1 = ufl.TrialFunctions(W)
        v0, v1 = ufl.TestFunctions(W)
        a = inner(u0, v0) * dx + inner(u1, v1) * dx
        L = inner(f, v0) * ufl.dx + inner(g, v1) * dx
        a, L = form(a), form(L)

        bdofsW0_V0 = locate_dofs_topological(W.sub(0), facetdim, bndry_facets)
        bdofsW1_V1 = locate_dofs_topological(W.sub(1), facetdim, bndry_facets)
        bcs = [dirichletbc(u0_bc, bdofsW0_V0, W.sub(0)), dirichletbc(u1_bc, bdofsW1_V1, W.sub(1))]

        A = petsc_assemble_matrix(a, bcs=bcs)
        A.assemble()
        b = petsc_assemble_vector(L)
        petsc_apply_lifting(b, [a], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc_set_bc(b, bcs)

        x = b.copy()
        ksp = PETSc.KSP()
        ksp.create(mesh.comm)
        ksp.setMonitor(monitor)
        ksp.setOperators(A)
        ksp.setType("cg")
        ksp.getPC().setType("jacobi")
        ksp.setTolerances(rtol=1.0e-12)
        ksp.setFromOptions()
        ksp.solve(b, x)
        Anorm = A.norm()
        bnorm = b.norm()
        xnorm = x.norm()
        ksp.destroy(), A.destroy(), b.destroy(), x.destroy()
        return Anorm, bnorm, xnorm

    Anorm0, bnorm0, xnorm0 = blocked()
    Anorm1, bnorm1, xnorm1 = nested()
    assert Anorm1 == pytest.approx(Anorm0, 1.0e-6)
    assert bnorm1 == pytest.approx(bnorm0, 1.0e-6)
    assert xnorm1 == pytest.approx(xnorm0, 1.0e-5)

    Anorm2, bnorm2, xnorm2 = monolithic()
    assert Anorm2 == pytest.approx(Anorm0, 1.0e-6)
    assert bnorm2 == pytest.approx(bnorm0, 1.0e-6)
    assert xnorm2 == pytest.approx(xnorm0, 1.0e-6)


@pytest.mark.parametrize(
    "mesh",
    [
        create_unit_square(MPI.COMM_WORLD, 12, 11, ghost_mode=GhostMode.none),
        create_unit_square(MPI.COMM_WORLD, 12, 11, ghost_mode=GhostMode.shared_facet),
        create_unit_cube(MPI.COMM_WORLD, 3, 7, 3, ghost_mode=GhostMode.none),
        create_unit_cube(MPI.COMM_WORLD, 3, 7, 3, ghost_mode=GhostMode.shared_facet),
    ],
)
def test_assembly_solve_taylor_hood(mesh):
    """Assemble Stokes problem with Taylor-Hood elements and solve."""
    gdim = mesh.geometry.dim
    P2 = functionspace(mesh, ("Lagrange", 2, (gdim,)))
    P1 = functionspace(mesh, ("Lagrange", 1))

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
        A = petsc_assemble_matrix_nest(
            form([[a00, a01], [a10, a11]]), bcs=[bc0, bc1], mat_types=[["baij", "aij"], ["aij", ""]]
        )
        A.assemble()
        P = petsc_assemble_matrix_nest(
            form([[p00, p01], [p10, p11]]), bcs=[bc0, bc1], mat_types=[["aij", "aij"], ["aij", ""]]
        )
        P.assemble()
        b = petsc_assemble_vector_nest(form([L0, L1]))
        petsc_apply_lifting_nest(b, form([[a00, a01], [a10, a11]]), [bc0, bc1])
        for b_sub in b.getNestSubVecs():
            b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        bcs = bcs_by_block(extract_function_spaces(form([L0, L1])), [bc0, bc1])
        petsc_set_bc_nest(b, bcs)
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
        ksp_u.getPC().setType("lu")
        ksp_p.setType("preonly")

        def monitor(ksp, its, rnorm):
            pass

        ksp.setTolerances(rtol=1.0e-8, max_it=50)
        ksp.setMonitor(monitor)
        ksp.setFromOptions()
        x = b.copy()
        ksp.solve(b, x)
        assert ksp.getConvergedReason() > 0
        norms = (b.norm(), x.norm(), nest_matrix_norm(A), nest_matrix_norm(P))
        pc.destroy(), ksp.destroy()
        A.destroy()
        b.destroy(), x.destroy()
        return norms

    def blocked_solve():
        """Blocked (monolithic) solver"""
        A = petsc_assemble_matrix_block(form([[a00, a01], [a10, a11]]), bcs=[bc0, bc1])
        A.assemble()
        P = petsc_assemble_matrix_block(form([[p00, p01], [p10, p11]]), bcs=[bc0, bc1])
        P.assemble()
        b = petsc_assemble_vector_block(
            form([L0, L1]), form([[a00, a01], [a10, a11]]), bcs=[bc0, bc1]
        )

        ksp = PETSc.KSP()
        ksp.create(mesh.comm)
        ksp.setOperators(A, P)
        ksp.setType("minres")
        pc = ksp.getPC()
        pc.setType("lu")
        ksp.setTolerances(rtol=1.0e-8, max_it=50)
        ksp.setFromOptions()
        x = A.createVecRight()
        ksp.solve(b, x)
        assert ksp.getConvergedReason() > 0
        ksp.destroy()
        return b.norm(), x.norm(), A.norm(), P.norm()

    def monolithic_solve():
        """Monolithic (interleaved) solver"""
        P2_el = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
        P1_el = element("Lagrange", mesh.basix_cell(), 1)
        TH = mixed_element([P2_el, P1_el])
        W = functionspace(mesh, TH)
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

        A = petsc_assemble_matrix(a, bcs=[bc0, bc1])
        A.assemble()
        P = petsc_assemble_matrix(p_form, bcs=[bc0, bc1])
        P.assemble()

        b = petsc_assemble_vector(L)
        petsc_apply_lifting(b, [a], bcs=[[bc0, bc1]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc_set_bc(b, [bc0, bc1])

        ksp = PETSc.KSP()
        ksp.create(mesh.comm)
        ksp.setOperators(A, P)
        ksp.setType("minres")
        pc = ksp.getPC()
        pc.setType("lu")

        def monitor(ksp, its, rnorm):
            # print("Num it, rnorm:", its, rnorm)
            pass

        ksp.setTolerances(rtol=1.0e-8, max_it=100)
        ksp.setMonitor(monitor)
        ksp.setFromOptions()
        x = A.createVecRight()
        ksp.solve(b, x)
        assert ksp.getConvergedReason() > 0
        ksp.destroy()
        return b.norm(), x.norm(), A.norm(), P.norm()

    bnorm0, xnorm0, Anorm0, Pnorm0 = nested_solve()
    bnorm1, xnorm1, Anorm1, Pnorm1 = blocked_solve()
    assert bnorm1 == pytest.approx(bnorm0, 1.0e-6)
    assert xnorm1 == pytest.approx(xnorm0, 1.0e-5)
    assert Anorm1 == pytest.approx(Anorm0, 1.0e-4)
    assert Pnorm1 == pytest.approx(Pnorm0, 1.0e-6)

    bnorm2, xnorm2, Anorm2, Pnorm2 = monolithic_solve()
    assert bnorm2 == pytest.approx(bnorm1, 1.0e-6)
    assert xnorm2 == pytest.approx(xnorm1, 1.0e-5)
    assert Anorm2 == pytest.approx(Anorm1, 1.0e-4)
    assert Pnorm2 == pytest.approx(Pnorm1, 1.0e-6)


def test_basic_interior_facet_assembly():
    mesh = create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [5, 5],
        cell_type=CellType.triangle,
        ghost_mode=GhostMode.shared_facet,
    )
    V = functionspace(mesh, ("DG", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(ufl.avg(u), ufl.avg(v)) * ufl.dS
    a = form(a)
    A = petsc_assemble_matrix(a)
    A.assemble()
    assert isinstance(A, PETSc.Mat)
    L = ufl.conj(ufl.avg(v)) * ufl.dS
    L = form(L)
    b = petsc_assemble_vector(L)
    b.assemble()
    assert isinstance(b, PETSc.Vec)
    A.destroy()
    b.destroy()


@pytest.mark.parametrize(
    "mesh",
    [
        create_unit_square(MPI.COMM_WORLD, 5, 5, ghost_mode=GhostMode.shared_facet),
        create_unit_cube(MPI.COMM_WORLD, 5, 5, 5, ghost_mode=GhostMode.shared_facet),
    ],
)
def test_symmetry_interior_facet_assembly(mesh):
    def bc(V):
        facetdim = mesh.topology.dim - 1
        bndry_facets = locate_entities_boundary(mesh, facetdim, lambda x: np.isclose(x[0], 0.0))
        bdofsV = locate_dofs_topological(V, facetdim, bndry_facets)
        u_bc = Function(V)
        return dirichletbc(u_bc, bdofsV)

    V0 = functionspace(mesh, ("N2E", 2))
    V1 = functionspace(mesh, ("RT", 3))
    u0, v0 = ufl.TrialFunction(V0), ufl.TestFunction(V0)
    u1, v1 = ufl.TrialFunction(V1), ufl.TestFunction(V1)
    a00 = inner(u0, v0) * dx
    a11 = inner(u1, v1) * dx
    a01 = inner(ufl.avg(u1), ufl.avg(v0)) * dS
    a10 = inner(ufl.avg(u0), ufl.avg(v1)) * dS
    a = form([[a00, a01], [a10, a11]])
    L0 = inner(ufl.unit_vector(0, mesh.geometry.dim), ufl.avg(v0)) * dS
    L1 = inner(ufl.unit_vector(1, mesh.geometry.dim), ufl.avg(v1)) * dS
    L = form([L0, L1])
    # without boundary conditions
    A = petsc_assemble_matrix_block(a)
    A.assemble()
    assert isinstance(A, PETSc.Mat)
    assert A.isSymmetric(tol=1.0e-4)
    A.destroy()
    # with boundary conditions
    bcs = [bc(V0), bc(V1)]
    A = petsc_assemble_matrix_block(a, bcs=bcs)
    b = petsc_assemble_vector_block(L, a, bcs=bcs)
    A.assemble()
    b.assemble()
    assert isinstance(A, PETSc.Mat)
    assert isinstance(b, PETSc.Vec)
    assert A.isSymmetric(tol=1.0e-4)
    A.destroy()
    b.destroy()

    V0 = functionspace(mesh, ("N2E", 1))
    V1 = functionspace(mesh, ("Regge", 1))
    u0, v0 = ufl.TrialFunction(V0), ufl.TestFunction(V0)
    u1, v1 = ufl.TrialFunction(V1), ufl.TestFunction(V1)
    n = ufl.FacetNormal(mesh)
    a00 = inner(u0, v0) * dx
    a11 = inner(u1, v1) * dx
    a01 = inner(ufl.dot(ufl.avg(u1), n("+")), ufl.avg(v0)) * dS
    a10 = inner(ufl.avg(u0), ufl.dot(ufl.avg(v1), n("+"))) * dS
    a = form([[a00, a01], [a10, a11]])
    L0 = inner(ufl.unit_vector(0, mesh.geometry.dim), ufl.avg(v0)) * dS
    L1 = inner(ufl.unit_matrix(1, 1, mesh.geometry.dim), ufl.avg(v1)) * dS
    L = form([L0, L1])
    # without boundary conditions
    A = petsc_assemble_matrix_block(a)
    A.assemble()
    assert isinstance(A, PETSc.Mat)
    assert A.isSymmetric(tol=1.0e-4)
    A.destroy()
    # with boundary conditions
    bcs = [bc(V0), bc(V1)]
    A = petsc_assemble_matrix_block(a, bcs=bcs)
    b = petsc_assemble_vector_block(L, a, bcs=bcs)
    A.assemble()
    b.assemble()
    assert isinstance(A, PETSc.Mat)
    assert isinstance(b, PETSc.Vec)
    assert A.isSymmetric(tol=1.0e-4)
    A.destroy()
    b.destroy()


@pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_basic_assembly_constant(mode, dtype):
    """Tests assembly with Constant.

    The following test should be sensitive to order of flattening the
    matrix-valued constant.
    """
    xtype = dtype(0).real.dtype
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5, ghost_mode=mode, dtype=xtype)
    V = functionspace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    c = Constant(mesh, np.array([[1.0, 2.0], [5.0, 3.0]], dtype=dtype))

    a = inner(c[1, 0] * u, v) * dx + inner(c[1, 0] * u, v) * ds
    L = inner(c[1, 0], v) * dx + inner(c[1, 0], v) * ds
    a, L = form(a, dtype=dtype), form(L, dtype=dtype)

    # Initial assembly
    A1 = fem.assemble_matrix(a)
    A1.scatter_reverse()

    b1 = fem.assemble_vector(L)
    b1.scatter_reverse(la.InsertMode.add)

    c.value = [[1.0, 2.0], [3.0, 4.0]]

    A2 = fem.assemble_matrix(a)
    A2.scatter_reverse()
    assert np.linalg.norm(A1.data * 3.0 - A2.data * 5.0) == pytest.approx(0.0, abs=1.0e-5)

    b2 = fem.assemble_vector(L)
    b2.scatter_reverse(la.InsertMode.add)
    assert np.linalg.norm(b1.array * 3.0 - b2.array * 5.0) == pytest.approx(0.0, abs=1.0e-5)


def test_lambda_assembler():
    """Tests assembly with a lambda function"""
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5)
    V = functionspace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    a = inner(u, v) * dx

    # Initial assembly
    a_form = form(a)

    rdata = []
    cdata = []
    vdata = []

    def mat_insert(rows, cols, vals):
        vdata.append(list(vals))
        rdata.append(list(np.repeat(rows, len(cols))))
        cdata.append(list(np.tile(cols, len(rows))))
        return 0

    _cpp.fem.assemble_matrix(mat_insert, a_form._cpp_object, [])
    vdata = np.array(vdata).flatten()
    cdata = np.array(cdata).flatten()
    rdata = np.array(rdata).flatten()
    mat = scipy.sparse.coo_matrix((vdata, (rdata, cdata)))
    v = np.ones(mat.shape[1])
    s = MPI.COMM_WORLD.allreduce(mat.dot(v).sum(), MPI.SUM)
    assert np.isclose(s, 1.0)


def test_pack_coefficients():
    """Test packing of form coefficients ahead of main assembly call."""
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 15)
    V = functionspace(mesh, ("Lagrange", 1))

    # Non-blocked
    u = Function(V)
    v = ufl.TestFunction(V)
    c = Constant(mesh, PETSc.ScalarType(12.0))
    F = ufl.inner(c, v) * dx - c * ufl.sqrt(u * u) * ufl.inner(u, v) * dx
    u.x.array[:] = 10.0
    _F = form(F)

    # -- Test vector
    b0 = petsc_assemble_vector(_F)
    b0.assemble()
    constants = _cpp.fem.pack_constants(_F._cpp_object)
    coeffs = _cpp.fem.pack_coefficients(_F._cpp_object)
    with b0.localForm() as _b0:
        for c in [(None, None), (None, coeffs), (constants, None), (constants, coeffs)]:
            b = petsc_assemble_vector(_F, c[0], c[1])
            b.assemble()
            with b.localForm() as _b:
                assert (_b0.array_r == _b.array_r).all()

    # Change coefficients
    constants *= 5.0
    for coeff in coeffs.values():
        coeff *= 5.0
    with b0.localForm() as _b0:
        for c in [(None, coeffs), (constants, None), (constants, coeffs)]:
            b = petsc_assemble_vector(_F, c[0], c[1])
            b.assemble()
            with b.localForm() as _b:
                assert (_b0 - _b).norm() > 1.0e-5

    # -- Test matrix
    du = ufl.TrialFunction(V)
    J = ufl.derivative(F, u, du)
    J = form(J)

    A0 = petsc_assemble_matrix(J)
    A0.assemble()

    constants = _cpp.fem.pack_constants(J._cpp_object)
    coeffs = _cpp.fem.pack_coefficients(J._cpp_object)
    for c in [(None, None), (None, coeffs), (constants, None), (constants, coeffs)]:
        A = petsc_assemble_matrix(J, constants=c[0], coeffs=c[1])
        A.assemble()
    assert 0.0 == pytest.approx((A - A0).norm(), abs=1.0e-12)  # /NOSONAR

    # Change coefficients
    constants *= 5.0
    for coeff in coeffs.values():
        coeff *= 5.0
    for c in [(None, coeffs), (constants, None), (constants, coeffs)]:
        A = petsc_assemble_matrix(J, constants=c[0], coeffs=c[1])
        A.assemble()
        assert (A - A0).norm() > 1.0e-5

    A.destroy(), A0.destroy()


def test_coefficents_non_constant():
    "Test packing coefficients with non-constant values"
    mesh = create_unit_square(MPI.COMM_WORLD, 3, 5)
    V = functionspace(mesh, ("Lagrange", 3))  # degree 3 so that interpolation is exact

    u = Function(V)
    u.interpolate(lambda x: x[0] * x[1] ** 2)
    x = SpatialCoordinate(mesh)

    v = ufl.TestFunction(V)

    # -- Volume integral vector
    F = form((ufl.inner(u, v) - ufl.inner(x[0] * x[1] ** 2, v)) * dx)
    b0 = petsc_assemble_vector(F)
    b0.assemble()
    assert np.linalg.norm(b0.array) == pytest.approx(0.0, abs=1.0e-7)

    # -- Exterior facet integral vector
    F = form((ufl.inner(u, v) - ufl.inner(x[0] * x[1] ** 2, v)) * ds)
    b0 = petsc_assemble_vector(F)
    b0.assemble()
    assert np.linalg.norm(b0.array) == pytest.approx(0.0, abs=1.0e-7)

    # -- Interior facet integral vector
    V = functionspace(mesh, ("DG", 3))  # degree 3 so that interpolation is exact

    u0 = Function(V)
    u0.interpolate(lambda x: x[1] ** 2)
    u1 = Function(V)
    u1.interpolate(lambda x: x[0])
    x = SpatialCoordinate(mesh)

    v = ufl.TestFunction(V)

    F = (
        ufl.inner(u1("+") * u0("-"), ufl.avg(v)) - ufl.inner(x[0] * x[1] ** 2, ufl.avg(v))
    ) * ufl.dS
    F = form(F)
    b0 = petsc_assemble_vector(F)
    b0.assemble()
    assert np.linalg.norm(b0.array) == pytest.approx(0.0, abs=1.0e-7)

    b0.destroy()


def test_vector_types():
    """Assemble form using different types"""
    mesh0 = create_unit_square(MPI.COMM_WORLD, 3, 5, dtype=np.float32)
    mesh1 = create_unit_square(MPI.COMM_WORLD, 3, 5, dtype=np.float64)
    V0, V1 = functionspace(mesh0, ("Lagrange", 3)), functionspace(mesh1, ("Lagrange", 3))
    v0, v1 = ufl.TestFunction(V0), ufl.TestFunction(V1)

    c = Constant(mesh1, np.float64(1))
    L = inner(c, v1) * ufl.dx
    x0 = la.vector(V1.dofmap.index_map, V1.dofmap.index_map_bs, dtype=np.float64)
    L = form(L, dtype=x0.array.dtype)
    c0 = _cpp.fem.pack_constants(L._cpp_object)
    c1 = _cpp.fem.pack_coefficients(L._cpp_object)
    _cpp.fem.assemble_vector(x0.array, L._cpp_object, c0, c1)
    x0.scatter_reverse(la.InsertMode.add)

    c = Constant(mesh1, np.complex128(1))
    L = inner(c, v1) * ufl.dx
    x1 = la.vector(V1.dofmap.index_map, V1.dofmap.index_map_bs, dtype=np.complex128)
    L = form(L, dtype=x1.array.dtype)
    c0 = _cpp.fem.pack_constants(L._cpp_object)
    c1 = _cpp.fem.pack_coefficients(L._cpp_object)
    _cpp.fem.assemble_vector(x1.array, L._cpp_object, c0, c1)
    x1.scatter_reverse(la.InsertMode.add)

    c = Constant(mesh0, np.float32(1))
    L = inner(c, v0) * ufl.dx
    x2 = la.vector(V0.dofmap.index_map, V0.dofmap.index_map_bs, dtype=np.float32)
    L = form(L, dtype=x2.array.dtype)
    c0 = _cpp.fem.pack_constants(L._cpp_object)
    c1 = _cpp.fem.pack_coefficients(L._cpp_object)
    _cpp.fem.assemble_vector(x2.array, L._cpp_object, c0, c1)
    x2.scatter_reverse(la.InsertMode.add)

    assert np.linalg.norm(x0.array - x1.array) == pytest.approx(0.0)
    assert np.linalg.norm(x0.array - x2.array) == pytest.approx(0.0, abs=1e-7)


def test_assemble_empty_rank_mesh():
    """Assembly on mesh where some ranks are empty"""
    comm = MPI.COMM_WORLD
    cell_type = CellType.triangle
    domain = ufl.Mesh(element("Lagrange", cell_type.name, 1, shape=(2,), dtype=default_real_type))

    def partitioner(comm, nparts, local_graph, num_ghost_nodes):
        """Leave cells on the curent rank"""
        dest = np.full(len(cells), comm.rank, dtype=np.int32)
        return graph.adjacencylist(dest)

    if comm.rank == 0:
        # Put cells on rank 0
        cells = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        x = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=default_real_type)
    else:
        # No cells on other ranks
        cells = np.empty((0, 3), dtype=np.int64)
        x = np.empty((0, 2), dtype=default_real_type)

    mesh = create_mesh(comm, cells, x, domain, partitioner)

    V = functionspace(mesh, ("Lagrange", 2))
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
    A = petsc_assemble_matrix(a)
    A.assemble()
    b = petsc_assemble_vector(L)
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
    ksp.destroy(), b.destroy(), A.destroy()


@pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
def test_matrix_assembly_rectangular(mode):
    """Test assembly of block rectangular block matrices"""
    msh = create_unit_square(MPI.COMM_WORLD, 4, 8, ghost_mode=mode)
    V0 = functionspace(msh, ("Lagrange", 1))
    V1 = V0.clone()
    u = ufl.TrialFunction(V0)
    v0, v1 = ufl.TestFunction(V0), ufl.TestFunction(V1)

    def single():
        a = form(ufl.inner(u, v0) * ufl.dx)
        A = petsc_assemble_matrix(a, bcs=[])
        A.assemble()
        return A

    def block():
        a = form([[ufl.inner(u, v0) * ufl.dx], [ufl.inner(u, v1) * ufl.dx]])
        A0 = petsc_assemble_matrix_block(a, bcs=[])
        A0.assemble()
        A1 = petsc_assemble_matrix_nest(a, bcs=[])
        A1.assemble()
        return A0, A1

    A0 = single()
    A1, A2 = block()
    assert A1.norm() == pytest.approx(np.sqrt(2) * A0.norm(), rel=1.0e-6, abs=1.0e-6)
    for row in range(2):
        A_sub = A2.getNestSubMatrix(row, 0)
        assert A_sub.equal(A0)

    A0.destroy(), A1.destroy(), A2.destroy()
