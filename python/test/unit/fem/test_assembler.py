# Copyright (C) 2018-2025 Garth N. Wells, Jørgen S. Dokken and Paul T. Kühner
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly"""

import math

from mpi4py import MPI

import numpy as np
import pytest
import scipy.sparse

import basix
import dolfinx.cpp
import ufl
from basix.ufl import element, mixed_element
from dolfinx import cpp as _cpp
from dolfinx import default_real_type, default_scalar_type, fem, graph, la, mesh
from dolfinx.fem import (
    Constant,
    Function,
    assemble_matrix,
    assemble_scalar,
    assemble_vector,
    bcs_by_block,
    dirichletbc,
    extract_function_spaces,
    form,
    functionspace,
    locate_dofs_geometrical,
    locate_dofs_topological,
    pack_coefficients,
    pack_constants,
)
from dolfinx.mesh import (
    CellType,
    GhostMode,
    create_mesh,
    create_rectangle,
    create_unit_cube,
    create_unit_square,
    exterior_facet_indices,
    locate_entities,
    locate_entities_boundary,
    meshtags,
)
from ufl import derivative, dS, ds, dx, inner
from ufl.geometry import SpatialCoordinate

dtype_parametrize = pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        pytest.param(np.complex64, marks=pytest.mark.xfail_win32_complex),
        pytest.param(np.complex128, marks=pytest.mark.xfail_win32_complex),
    ],
)


@pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
@dtype_parametrize
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
@dtype_parametrize
def test_assemble_functional_ds(mode, dtype):
    xtype = dtype(0).real.dtype
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, ghost_mode=mode, dtype=xtype)
    M = form(1.0 * ds(domain=mesh), dtype=dtype)
    value = assemble_scalar(M)
    value = mesh.comm.allreduce(value, op=MPI.SUM)
    assert value == pytest.approx(4.0, 1e-6)


@dtype_parametrize
def test_assemble_derivatives(dtype):
    """This test checks the original_coefficient_positions, which may change
    under differentiation (some coefficients and constants are
    eliminated)."""
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

    A1 = assemble_matrix(a)
    A1.scatter_reverse()
    a = form(c2 * b * inner(du, v) * dx, dtype=dtype)
    A2 = assemble_matrix(a)
    A2.scatter_reverse()
    assert np.allclose(A1.data, A2.data)


@pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
@dtype_parametrize
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
    A = assemble_matrix(a)
    A.scatter_reverse()
    assert isinstance(A, la.MatrixCSR)
    b = assemble_vector(L)
    b.scatter_reverse(la.InsertMode.add)
    assert isinstance(b, la.Vector)

    # Second assembly
    normA = A.squared_norm()
    A.set_value(0)
    A = assemble_matrix(A, a)
    A.scatter_reverse()
    assert isinstance(A, la.MatrixCSR)
    assert normA == pytest.approx(A.squared_norm())
    normb = la.norm(b)
    b.array[:] = 0
    assemble_vector(b.array, L)
    b.scatter_reverse(la.InsertMode.add)
    assert normb == pytest.approx(la.norm(b))

    # Vector re-assembly - no zeroing (but need to zero ghost entries)
    b.array[b.index_map.size_local * b.block_size :] = 0
    assemble_vector(b.array, L)
    b.scatter_reverse(la.InsertMode.add)
    assert 2 * normb == pytest.approx(la.norm(b))

    # Matrix re-assembly (no zeroing)
    assemble_matrix(A, a)
    A.scatter_reverse()
    assert 4 * normA == pytest.approx(A.squared_norm())


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


@pytest.mark.petsc4py
class TestPETScAssemblers:
    @pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
    def test_basic_assembly_petsc_matrixcsr(self, mode):
        from petsc4py import PETSc

        from dolfinx.fem.petsc import assemble_matrix as petsc_assemble_matrix

        mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, ghost_mode=mode)
        V = functionspace(mesh, ("Lagrange", 1))
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        a = form(inner(u, v) * dx + inner(u, v) * ds)

        A0 = assemble_matrix(a)
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
        A0 = assemble_matrix(a)
        A0.scatter_reverse()
        assert isinstance(A0, la.MatrixCSR)
        A1 = petsc_assemble_matrix(a)
        A1.assemble()
        assert isinstance(A1, PETSc.Mat)
        assert np.sqrt(A0.squared_norm()) == pytest.approx(A1.norm(), rel=1.0e-8, abs=1.0e-5)
        A1.destroy()

    @pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
    def test_assembly_bcs(self, mode):
        from petsc4py import PETSc

        from dolfinx.fem.petsc import apply_lifting as petsc_apply_lifting
        from dolfinx.fem.petsc import assemble_matrix as petsc_assemble_matrix
        from dolfinx.fem.petsc import assemble_vector as petsc_assemble_vector
        from dolfinx.fem.petsc import set_bc as petsc_set_bc

        mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, ghost_mode=mode)
        V = functionspace(mesh, ("Lagrange", 1))
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        a = form(inner(u, v) * dx + inner(u, v) * ds)
        L = form(inner(1.0, v) * dx)

        bdofsV = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0))
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
    def test_petsc_assemble_manifold(self):
        """Test assembly of poisson problem on a mesh with topological
        dimension 1 but embedded in 2D (gdim=2).
        """
        from petsc4py import PETSc

        from dolfinx.fem.petsc import apply_lifting as petsc_apply_lifting
        from dolfinx.fem.petsc import assemble_matrix as petsc_assemble_matrix
        from dolfinx.fem.petsc import assemble_vector as petsc_assemble_vector
        from dolfinx.fem.petsc import set_bc as petsc_set_bc

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
        mesh = create_mesh(MPI.COMM_WORLD, cells, domain, points)
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
    def test_matrix_assembly_block(self, mode):
        """Test assembly of block matrices and vectors into (a) monolithic
        blocked structures, PETSc Nest structures, and monolithic
        structures.
        """
        from petsc4py import PETSc

        from dolfinx.fem.petsc import apply_lifting as petsc_apply_lifting
        from dolfinx.fem.petsc import assemble_matrix as petsc_assemble_matrix
        from dolfinx.fem.petsc import assemble_vector as petsc_assemble_vector
        from dolfinx.fem.petsc import set_bc as petsc_set_bc

        mesh = create_unit_square(MPI.COMM_WORLD, 4, 8, ghost_mode=mode)
        p0, p1 = 1, 2
        P0 = element("Lagrange", mesh.basix_cell(), p0, dtype=default_real_type)
        P1 = element("Lagrange", mesh.basix_cell(), p1, dtype=default_real_type)
        P2 = element("Lagrange", mesh.basix_cell(), p0, dtype=default_real_type)
        V0 = functionspace(mesh, P0)
        V1 = functionspace(mesh, P1)
        V2 = functionspace(mesh, P2)

        # Locate facets on boundary
        facetdim = mesh.topology.dim - 1
        bndry_facets = locate_entities_boundary(
            mesh, facetdim, lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)
        )
        bdofsV1 = locate_dofs_topological(V1, facetdim, bndry_facets)
        u_bc = PETSc.ScalarType(50.0)
        bc = dirichletbc(u_bc, bdofsV1, V1)

        # Define variational problem
        u, p, r = ufl.TrialFunction(V0), ufl.TrialFunction(V1), ufl.TrialFunction(V2)
        v, q, s = ufl.TestFunction(V0), ufl.TestFunction(V1), ufl.TestFunction(V2)
        g = -3.0

        a00 = inner(u, v) * dx
        a01 = inner(p, v) * dx
        a02 = inner(r, v) * dx
        a10 = inner(u, q) * dx
        a11 = inner(p, q) * dx
        a12 = inner(r, q) * dx
        a20 = inner(u, s) * dx
        a21 = inner(p, s) * dx
        a22 = inner(r, s) * dx

        L0 = ufl.ZeroBaseForm((v,))
        L1 = inner(g, q) * dx
        L2 = inner(g, s) * dx

        a_block = form([[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]])
        L_block = form([L0, L1, L2])

        # Prepare a block problem with "None" on (1, 1) diagonal
        a_block_none = form([[a00, a01, a02], [None, None, a12], [a20, a21, a22]])

        def blocked():
            """Monolithic blocked"""
            A = petsc_assemble_matrix(a_block, bcs=[bc])
            A.assemble()
            b = petsc_assemble_vector(L_block, kind=PETSc.Vec.Type.MPI)
            bcs1 = fem.bcs_by_block(fem.extract_function_spaces(a_block, 1), bcs=[bc])
            petsc_apply_lifting(b, a_block, bcs=bcs1)
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            bcs0 = fem.bcs_by_block(fem.extract_function_spaces(L_block), [bc])
            petsc_set_bc(b, bcs0)

            assert A.getType() != "nest"

            with pytest.raises(RuntimeError):
                petsc_assemble_matrix(a_block_none, bcs=[bc])

            return A, b

        def nest():
            """Nested (MatNest)"""
            A = petsc_assemble_matrix(
                a_block,
                bcs=[bc],
                kind=[["baij", "aij", "aij"], ["aij", "", "aij"], ["aij", "aij", "aij"]],
            )
            A.assemble()
            assert A.type == "nest"

            with pytest.raises(RuntimeError):
                petsc_assemble_matrix(a_block_none, bcs=[bc])

            b = petsc_assemble_vector(L_block, kind="nest")
            bcs1 = fem.bcs_by_block(fem.extract_function_spaces(a_block, 1), bcs=[bc])
            petsc_apply_lifting(b, a_block, bcs=bcs1)
            for b_sub in b.getNestSubVecs():
                b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            bcs0 = bcs_by_block([L.function_spaces[0] for L in L_block], [bc])
            petsc_set_bc(b, bcs0)
            b.assemble()

            return A, b

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
            L = ufl.ZeroBaseForm((v0,)) + inner(g, v1) * dx + inner(g, v2) * dx
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

            return A, b

        A_blocked, b_blocked = blocked()
        A_nest, b_nest = nest()
        A_monolithic, b_monolithic = monolithic()

        assert A_blocked.equal(A_nest)
        assert b_blocked.equal(b_nest)

        assert A_blocked.norm() == pytest.approx(A_monolithic.norm(), 1.0e-4)
        assert b_blocked.norm() == pytest.approx(b_monolithic.norm(), 1.0e-6)

        A_nest.destroy(), b_nest.destroy()
        A_blocked.destroy(), b_blocked.destroy()
        A_monolithic.destroy(), b_monolithic.destroy()

    @pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
    def test_matrix_assembly_block_vector(self, mode):
        """Test assembly of block matrices and vectors into (a) monolithic
        blocked structures, PETSc Nest structures, and monolithic
        structures.
        """
        from petsc4py import PETSc

        from dolfinx.fem.petsc import apply_lifting as petsc_apply_lifting
        from dolfinx.fem.petsc import assemble_matrix as petsc_assemble_matrix
        from dolfinx.fem.petsc import assemble_vector as petsc_assemble_vector
        from dolfinx.fem.petsc import set_bc as petsc_set_bc

        mesh = create_unit_square(MPI.COMM_WORLD, 4, 8, ghost_mode=mode)

        P0 = element("Lagrange", mesh.basix_cell(), 2, dtype=default_real_type, shape=(2,))
        P1 = element("Lagrange", mesh.basix_cell(), 1, dtype=default_real_type)
        V0 = functionspace(mesh, P0)
        V1 = functionspace(mesh, P1)

        # Locate facets on boundary
        facetdim = mesh.topology.dim - 1
        bndry_facets = locate_entities_boundary(
            mesh, facetdim, lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)
        )
        bdofsV1 = locate_dofs_topological(V1, facetdim, bndry_facets)
        u_bc = PETSc.ScalarType(50.0)
        bc = dirichletbc(u_bc, bdofsV1, V1)

        # Define variational problem
        u, p = ufl.TrialFunction(V0), ufl.TrialFunction(V1)
        v, q = ufl.TestFunction(V0), ufl.TestFunction(V1)
        g = -3.0

        a00 = inner(u, v) * dx
        a01 = inner(p, v[0] + v[1]) * dx
        a10 = inner(u[0] + u[1], q) * dx
        a11 = inner(p, q) * dx

        L0 = ufl.ZeroBaseForm((v,))
        L1 = inner(g, q) * dx

        a_block = form([[a00, a01], [a10, a11]])
        L_block = form([L0, L1])

        def blocked():
            """Monolithic blocked"""
            A = petsc_assemble_matrix(a_block, bcs=[bc])
            A.assemble()
            b = petsc_assemble_vector(L_block, kind=PETSc.Vec.Type.MPI)
            bcs1 = fem.bcs_by_block(fem.extract_function_spaces(a_block, 1), bcs=[bc])
            petsc_apply_lifting(b, a_block, bcs=bcs1)
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            bcs0 = fem.bcs_by_block(fem.extract_function_spaces(L_block), [bc])
            petsc_set_bc(b, bcs0)

            assert A.getType() != "nest"
            return A, b

        def nest():
            """Nested (MatNest)"""
            A = petsc_assemble_matrix(a_block, bcs=[bc], kind="nest")
            A.assemble()

            assert A.type == "nest"

            b = petsc_assemble_vector(L_block, kind="nest")
            bcs1 = fem.bcs_by_block(fem.extract_function_spaces(a_block, 1), bcs=[bc])
            petsc_apply_lifting(b, a_block, bcs=bcs1)
            for b_sub in b.getNestSubVecs():
                b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            bcs0 = bcs_by_block([L.function_spaces[0] for L in L_block], [bc])
            petsc_set_bc(b, bcs0)
            b.assemble()

            return A, b

        def monolithic():
            """Monolithic version"""
            W = functionspace(mesh, mixed_element([P0, P1]))
            u0, u1 = ufl.TrialFunctions(W)
            v0, v1 = ufl.TestFunctions(W)
            a = (
                inner(u0, v0) * dx
                + inner(u1, v1) * dx
                + inner(u0[0] + u0[1], v1) * dx
                + inner(u1, v0[0] + v0[1]) * dx
            )
            L = ufl.ZeroBaseForm((v0,)) + inner(g, v1) * dx
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

            return A, b

        A_blocked, b_blocked = blocked()
        A_nest, b_nest = nest()
        A_monolithic, b_monolithic = monolithic()

        assert A_blocked.equal(A_nest)
        assert b_blocked.equal(b_nest)

        assert A_blocked.norm() == pytest.approx(A_monolithic.norm(), 1.0e-4)
        assert b_blocked.norm() == pytest.approx(b_monolithic.norm(), 1.0e-6)

        A_nest.destroy(), b_nest.destroy()
        A_blocked.destroy(), b_blocked.destroy()
        A_monolithic.destroy(), b_monolithic.destroy()

    @pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
    def test_assembly_solve_block(self, mode):
        """Solve a two-field mass-matrix like problem with block matrix approaches
        and test that solution is the same.
        """
        from petsc4py import PETSc

        from dolfinx.fem.petsc import apply_lifting as petsc_apply_lifting
        from dolfinx.fem.petsc import assemble_matrix as petsc_assemble_matrix
        from dolfinx.fem.petsc import assemble_vector as petsc_assemble_vector
        from dolfinx.fem.petsc import set_bc as petsc_set_bc

        mesh = create_unit_square(MPI.COMM_WORLD, 32, 31, ghost_mode=mode)
        P = element("Lagrange", mesh.basix_cell(), 1, dtype=default_real_type)
        V0 = functionspace(mesh, P)
        V1 = V0.clone()

        # Locate facets on boundary
        facetdim = mesh.topology.dim - 1
        bndry_facets = locate_entities_boundary(
            mesh, facetdim, lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)
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

        a00 = form(inner(u, v) * dx)
        a01 = None
        a10 = None
        a11 = form(inner(p, q) * dx)
        L0 = form(inner(f, v) * dx)
        L1 = form(inner(g, q) * dx)

        def monitor(ksp, its, rnorm):
            pass
            # print("Norm:", its, rnorm)

        def blocked():
            """Blocked"""
            a = [[a00, a01], [a10, a11]]
            A = petsc_assemble_matrix(a, bcs=bcs)
            b = petsc_assemble_vector([L0, L1], kind=PETSc.Vec.Type.MPI)
            bcs1 = fem.bcs_by_block(fem.extract_function_spaces(a, 1), bcs=bcs)
            petsc_apply_lifting(b, a, bcs=bcs1)
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            bcs0 = fem.bcs_by_block(fem.extract_function_spaces([L0, L1]), bcs)
            petsc_set_bc(b, bcs0)

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
            a = [[a00, a01], [a10, a11]]
            A = petsc_assemble_matrix(a, bcs=bcs, diag=1.0, kind="nest")
            A.assemble()
            b = petsc_assemble_vector([L0, L1], kind="nest")
            bcs1 = fem.bcs_by_block(fem.extract_function_spaces(a, 1), bcs=bcs)
            petsc_apply_lifting(b, a, bcs=bcs1)
            for b_sub in b.getNestSubVecs():
                b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            bcs0 = bcs_by_block([L0.function_spaces[0], L1.function_spaces[0]], bcs)
            petsc_set_bc(b, bcs0)
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
            bcs = [
                dirichletbc(u0_bc, bdofsW0_V0, W.sub(0)),
                dirichletbc(u1_bc, bdofsW1_V1, W.sub(1)),
            ]

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
        assert xnorm2 == pytest.approx(xnorm0, 1.0e-5)

    @pytest.mark.parametrize(
        "mesh",
        [
            create_unit_square(MPI.COMM_WORLD, 12, 11, ghost_mode=GhostMode.none),
            create_unit_square(MPI.COMM_WORLD, 12, 11, ghost_mode=GhostMode.shared_facet),
            create_unit_cube(MPI.COMM_WORLD, 3, 7, 3, ghost_mode=GhostMode.none),
            create_unit_cube(MPI.COMM_WORLD, 3, 7, 3, ghost_mode=GhostMode.shared_facet),
        ],
    )
    def test_assembly_solve_taylor_hood(self, mesh):
        """Assemble Stokes problem with Taylor-Hood elements and solve."""
        from petsc4py import PETSc

        from dolfinx.fem.petsc import apply_lifting as petsc_apply_lifting
        from dolfinx.fem.petsc import assemble_matrix as petsc_assemble_matrix
        from dolfinx.fem.petsc import assemble_vector as petsc_assemble_vector
        from dolfinx.fem.petsc import set_bc as petsc_set_bc

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

        # We need a 'zero' form for the 'zero' part of L
        f = Function(P2)
        L0 = ufl.inner(f, v) * dx
        L1 = ufl.ZeroBaseForm((q,))

        def nested_solve():
            """Nested solver"""
            a = form([[a00, a01], [a10, a11]])
            L = form([L0, L1])
            A = petsc_assemble_matrix(a, bcs=[bc0, bc1], kind=[["baij", "aij"], ["aij", ""]])
            A.assemble()
            P = petsc_assemble_matrix(
                form([[p00, p01], [p10, p11]]), bcs=[bc0, bc1], kind=[["aij", "aij"], ["aij", ""]]
            )
            P.assemble()
            b = petsc_assemble_vector(L, kind="nest")

            bcs1 = fem.bcs_by_block(fem.extract_function_spaces(a, 1), [bc0, bc1])
            petsc_apply_lifting(b, a, bcs1)
            for b_sub in b.getNestSubVecs():
                b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            bcs0 = bcs_by_block(extract_function_spaces(L), [bc0, bc1])
            petsc_set_bc(b, bcs0)
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
            A = petsc_assemble_matrix(form([[a00, a01], [a10, a11]]), bcs=[bc0, bc1])
            A.assemble()
            P = petsc_assemble_matrix(form([[p00, p01], [p10, p11]]), bcs=[bc0, bc1])
            P.assemble()
            L, a = form([L0, L1]), form([[a00, a01], [a10, a11]])
            b = petsc_assemble_vector(L, kind=PETSc.Vec.Type.MPI)
            bcs1 = fem.bcs_by_block(fem.extract_function_spaces(a, 1), [bc0, bc1])
            petsc_apply_lifting(b, a, bcs=bcs1)
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            bcs0 = fem.bcs_by_block(fem.extract_function_spaces(L), bcs=[bc0, bc1])
            petsc_set_bc(b, bcs0)

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
            P2_el = element(
                "Lagrange",
                mesh.basix_cell(),
                2,
                shape=(mesh.geometry.dim,),
                dtype=default_real_type,
            )
            P1_el = element("Lagrange", mesh.basix_cell(), 1, dtype=default_real_type)
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
            L0 = inner(f, v) * dx
            L1 = ufl.ZeroBaseForm((q,))
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

    def test_basic_interior_facet_assembly(self):
        from petsc4py import PETSc

        from dolfinx.fem.petsc import assemble_matrix as petsc_assemble_matrix
        from dolfinx.fem.petsc import assemble_vector as petsc_assemble_vector

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
    def test_symmetry_interior_facet_assembly(self, mesh):
        from petsc4py import PETSc

        from dolfinx.fem.petsc import apply_lifting as petsc_apply_lifting
        from dolfinx.fem.petsc import assemble_matrix as petsc_assemble_matrix
        from dolfinx.fem.petsc import assemble_vector as petsc_assemble_vector
        from dolfinx.fem.petsc import set_bc as petsc_set_bc

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
        A = petsc_assemble_matrix(a)
        A.assemble()
        assert isinstance(A, PETSc.Mat)
        assert A.isSymmetric(tol=1.0e-4)
        A.destroy()
        # with boundary conditions
        bcs = [bc(V0), bc(V1)]
        A = petsc_assemble_matrix(a, bcs=bcs)
        b = petsc_assemble_vector(L, kind=PETSc.Vec.Type.MPI)
        bcs1 = fem.bcs_by_block(fem.extract_function_spaces(a, 1), bcs)
        petsc_apply_lifting(b, a, bcs=bcs1)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        bcs0 = fem.bcs_by_block(fem.extract_function_spaces(L), bcs=bcs)
        petsc_set_bc(b, bcs0)

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
        A = petsc_assemble_matrix(a)
        A.assemble()
        assert isinstance(A, PETSc.Mat)
        assert A.isSymmetric(tol=1.0e-4)
        A.destroy()
        # with boundary conditions
        bcs = [bc(V0), bc(V1)]
        A = petsc_assemble_matrix(a, bcs=bcs)
        b = petsc_assemble_vector(L, kind=PETSc.Vec.Type.MPI)
        bcs1 = fem.bcs_by_block(fem.extract_function_spaces(a, 1), bcs=bcs)
        petsc_apply_lifting(b, a, bcs=bcs1)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        bcs0 = fem.bcs_by_block(fem.extract_function_spaces(L), bcs=bcs)
        petsc_set_bc(b, bcs0)

        A.assemble()
        b.assemble()
        assert isinstance(A, PETSc.Mat)
        assert isinstance(b, PETSc.Vec)
        assert A.isSymmetric(tol=1.0e-4)
        A.destroy()
        b.destroy()

    @pytest.mark.parametrize("kind", ["nest", "mpi", None])
    def test_pack_coefficients(self, kind):
        """Test packing of form coefficients ahead of main assembly call."""
        from petsc4py import PETSc

        from dolfinx.fem.petsc import assemble_matrix as petsc_assemble_matrix
        from dolfinx.fem.petsc import assemble_vector as petsc_assemble_vector

        mesh = create_unit_square(MPI.COMM_WORLD, 12, 15)

        if kind is None:
            # Non-blocked
            V = functionspace(mesh, ("Lagrange", 1))
            u = Function(V)
            v = ufl.TestFunction(V)
            c = Constant(mesh, PETSc.ScalarType(12.0))
            F = ufl.inner(c, v) * dx - c * ufl.sqrt(u * u) * ufl.inner(u, v) * dx
            u.x.array[:] = 10.0
            _F = form(F)
        else:
            V = functionspace(mesh, ("Lagrange", 1))
            V2 = V.clone()
            u = Function(V)
            u.interpolate(lambda x: x[0] * x[1])
            v = ufl.TestFunction(V)
            u2 = Function(V2)
            v2 = ufl.TestFunction(V2)
            c = Constant(mesh, PETSc.ScalarType(12.0))
            u2.interpolate(lambda x: x[0] + x[1])
            F = [c**2 * ufl.inner(u * u2, v2) * dx, c * ufl.inner(u * u2 * u2, v2) * dx]
            _F = form(F)

        # -- Test vector
        b0 = petsc_assemble_vector(_F, kind=kind)
        b0.assemble()
        constants = pack_constants(_F)
        coeffs = pack_coefficients(_F)
        with b0.localForm() as _b0:
            for c in [(None, None), (None, coeffs), (constants, None), (constants, coeffs)]:
                b = petsc_assemble_vector(_F, c[0], c[1], kind=kind)
                b.assemble()
                with b.localForm() as _b:
                    assert (_b0.array_r == _b.array_r).all()

        # Change coefficients and constants, check that it reflected in the vector
        if kind is None:
            constants *= 5.0
        else:
            for const in constants:
                const *= 5.0
        if kind is None:
            for coeff in coeffs.values():
                coeff *= 5.0
        else:
            for coeff in coeffs:
                for c in coeff.values():
                    c *= 5.0

        for c in [(constants, None), (None, coeffs), (constants, coeffs)]:
            b = petsc_assemble_vector(_F, c[0], c[1], kind=kind)
            b.assemble()
            diff = b0.copy()
            diff.axpy(-1.0, b)
            assert diff.norm() > 1.0e-5

        # -- Test matrix
        if kind is None:
            du = ufl.TrialFunction(V)
            _J = ufl.derivative(F, u, du)
        else:
            du = [ufl.TrialFunction(V), ufl.TrialFunction(V2)]
            us = [u, u2]
            _J = [
                [ufl.derivative(F[j], us[i], du[i]) for i in range(len(F))] for j in range(len(du))
            ]

        J = form(_J)
        A0 = petsc_assemble_matrix(J, kind=kind)
        A0.assemble()

        constants = pack_constants(J)
        coeffs = pack_coefficients(J)
        for c in [(None, None), (None, coeffs), (constants, None), (constants, coeffs)]:
            A = petsc_assemble_matrix(J, constants=c[0], coeffs=c[1], kind=kind)
            A.assemble()
            if kind == "nest":
                # Nest does not have norm implemented
                for i in range(2):
                    for j in range(2):
                        Asub = A.getNestSubMatrix(i, j)
                        A0sub = A0.getNestSubMatrix(i, j)
                        assert 0.0 == pytest.approx((Asub - A0sub).norm(), abs=1.0e-12)  # /NOSONAR
                        Asub.destroy()
                        A0sub.destroy()
            else:
                assert 0.0 == pytest.approx((A - A0).norm(), abs=1.0e-12)  # /NOSONAR

        # Change coefficients and constants
        if kind is None:
            constants *= 5.0
        else:
            for ci in constants:
                for cij in ci:
                    cij *= 5.0
        if kind is None:
            for coeff in coeffs.values():
                coeff *= 5.0
        else:
            for ci in coeffs:
                for cij in ci:
                    for c in cij.values():
                        c *= 5.0
        # Re-assemble with either new coefficients, new constants or both and check that
        # the matrix changes
        for c in [(None, coeffs), (constants, None), (constants, coeffs)]:
            A = petsc_assemble_matrix(J, constants=c[0], coeffs=c[1], kind=kind)
            A.assemble()
            if kind == "nest":
                # Nest does not have norm implemented
                for i in range(2):
                    for j in range(2):
                        Asub = A.getNestSubMatrix(i, j)
                        A0sub = A0.getNestSubMatrix(i, j)
                        assert (Asub - A0sub).norm() > 1.0e-5  # /NOSONAR
                        Asub.destroy()
                        A0sub.destroy()
            else:
                assert (A - A0).norm() > 1.0e-5  # /NOSONAR
            A.destroy()
        A0.destroy()

    @pytest.mark.parametrize("kind", ["nest", "mpi"])
    def test_lifting_coefficients(self, kind):
        from dolfinx.fem.petsc import apply_lifting as petsc_apply_lifting
        from dolfinx.fem.petsc import create_vector as petsc_create_vector

        mesh = create_unit_square(MPI.COMM_WORLD, 12, 15)
        V = functionspace(mesh, ("Lagrange", 1))
        Q = V.clone()
        k = Function(V)
        k.interpolate(lambda x: x[0] * x[1])
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        p = ufl.TrialFunction(Q)
        q = ufl.TestFunction(Q)

        # L = form([ufl.ZeroBaseForm((v,)), ufl.ZeroBaseForm((q,))])
        J = form(
            [[k * ufl.inner(u, v) * dx, None], [ufl.inner(u, q) * dx, k * ufl.inner(p, q) * dx]]
        )

        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        bndry_facets = exterior_facet_indices(mesh.topology)
        bndry_dofs = locate_dofs_topological(V, mesh.topology.dim - 1, bndry_facets)
        bcs = [dirichletbc(default_scalar_type(2.0), bndry_dofs, V)]

        bcs1 = bcs_by_block(extract_function_spaces(J, 1), bcs)

        # Apply lifting with input coefficient
        coeffs = pack_coefficients(J)
        b = petsc_create_vector([V, Q], kind=kind)
        assert b.equal(petsc_create_vector([V, Q]))
        petsc_apply_lifting(b, J, bcs=bcs1, coeffs=coeffs)
        b.assemble()

        # Reference lifting
        b_ref = petsc_create_vector([V, Q], kind=kind)
        petsc_apply_lifting(b_ref, J, bcs=bcs1)
        b_ref.assemble()

        np.testing.assert_allclose(b.array_r, b_ref.array_r, rtol=1e-12)

    def test_coefficents_non_constant(self):
        """Test packing coefficients with non-constant values."""
        from dolfinx.fem.petsc import assemble_vector as petsc_assemble_vector

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
        assert np.linalg.norm(b0.array) == pytest.approx(
            0.0, abs=np.sqrt(np.finfo(mesh.geometry.x.dtype).eps)
        )

        # -- Exterior facet integral vector
        F = form((ufl.inner(u, v) - ufl.inner(x[0] * x[1] ** 2, v)) * ds)
        b0 = petsc_assemble_vector(F)
        b0.assemble()
        assert np.linalg.norm(b0.array) == pytest.approx(
            0.0, abs=np.sqrt(np.finfo(mesh.geometry.x.dtype).eps)
        )

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
        assert np.linalg.norm(b0.array) == pytest.approx(
            0.0, abs=np.sqrt(np.finfo(mesh.geometry.x.dtype).eps)
        )

        b0.destroy()

    def test_assemble_empty_rank_mesh(self):
        """Assembly on mesh where some ranks are empty."""
        from petsc4py import PETSc

        from dolfinx.fem.petsc import assemble_matrix as petsc_assemble_matrix
        from dolfinx.fem.petsc import assemble_vector as petsc_assemble_vector

        comm = MPI.COMM_WORLD
        cell_type = CellType.triangle
        domain = ufl.Mesh(
            element("Lagrange", cell_type.name, 1, shape=(2,), dtype=default_real_type)
        )

        def partitioner(comm, nparts, local_graph, num_ghost_nodes):
            """Leave cells on the current rank."""
            dest = np.full(len(cells), comm.rank, dtype=np.int32)
            return graph.adjacencylist(dest)._cpp_object

        if comm.rank == 0:
            # Put cells on rank 0
            cells = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
            x = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=default_real_type)
        else:
            # No cells on other ranks
            cells = np.empty((0, 3), dtype=np.int64)
            x = np.empty((0, 2), dtype=default_real_type)

        mesh = create_mesh(comm, cells, domain, x, partitioner)

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
    def test_matrix_assembly_rectangular(self, mode):
        """Test assembly of block rectangular block matrices."""
        from dolfinx.fem.petsc import assemble_matrix as petsc_assemble_matrix

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
            A0 = petsc_assemble_matrix(a, bcs=[])
            A0.assemble()
            A1 = petsc_assemble_matrix(a, bcs=[], kind="nest")
            A1.assemble()
            return A0, A1

        A0 = single()
        A1, A2 = block()
        assert A1.norm() == pytest.approx(np.sqrt(2) * A0.norm(), rel=1.0e-6, abs=1.0e-6)
        for row in range(2):
            A_sub = A2.getNestSubMatrix(row, 0)
            assert A_sub.equal(A0)

        A0.destroy(), A1.destroy(), A2.destroy()

    def test_block_null_lifting(self):
        from petsc4py import PETSc

        from dolfinx.fem.petsc import assemble_vector as petsc_assemble_vector

        comm = MPI.COMM_WORLD
        msh = create_unit_square(comm, 2, 2)
        V = functionspace(msh, ("Lagrange", 1))
        W = functionspace(msh, ("Lagrange", 2))
        v, w = ufl.TestFunction(V), ufl.TestFunction(W)
        L = form([ufl.conj(v) * ufl.dx, ufl.conj(w) * ufl.dx])
        b = petsc_assemble_vector(L, kind=PETSc.Vec.Type.MPI)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    def test_zero_diagonal_block_no_bcs(self):
        from dolfinx.fem.petsc import assemble_matrix as petsc_assemble_matrix

        msh = create_unit_square(MPI.COMM_WORLD, 2, 2)
        V = functionspace(msh, ("Lagrange", 1))
        W = functionspace(msh, ("Lagrange", 2))
        u, p = ufl.TrialFunction(V), ufl.TrialFunction(W)
        v, q = ufl.TestFunction(V), ufl.TestFunction(W)
        a = form(
            [[ufl.inner(u, v) * ufl.dx, ufl.inner(p, v) * ufl.dx], [ufl.inner(u, q) * ufl.dx, None]]
        )
        A = petsc_assemble_matrix(a, kind="mpi")
        A.assemble()


@pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
@dtype_parametrize
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
    A1 = assemble_matrix(a)
    A1.scatter_reverse()

    b1 = assemble_vector(L)
    b1.scatter_reverse(la.InsertMode.add)

    c.value = [[1.0, 2.0], [3.0, 4.0]]

    A2 = assemble_matrix(a)
    A2.scatter_reverse()
    assert np.linalg.norm(A1.data * 3.0 - A2.data * 5.0) == pytest.approx(0.0, abs=1.0e-5)

    b2 = assemble_vector(L)
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


@pytest.mark.xfail_win32_complex
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
    c0 = pack_constants(L)
    c1 = pack_coefficients(L)
    assemble_vector(x0.array, L, c0, c1)
    x0.scatter_reverse(la.InsertMode.add)

    c = Constant(mesh1, np.complex128(1))
    L = inner(c, v1) * ufl.dx
    x1 = la.vector(V1.dofmap.index_map, V1.dofmap.index_map_bs, dtype=np.complex128)
    L = form(L, dtype=x1.array.dtype)
    c0 = pack_constants(L)
    c1 = pack_coefficients(L)
    assemble_vector(x1.array, L, c0, c1)
    x1.scatter_reverse(la.InsertMode.add)

    c = Constant(mesh0, np.float32(1))
    L = inner(c, v0) * ufl.dx
    x2 = la.vector(V0.dofmap.index_map, V0.dofmap.index_map_bs, dtype=np.float32)
    L = form(L, dtype=x2.array.dtype)
    c0 = pack_constants(L)
    c1 = pack_coefficients(L)
    assemble_vector(x2.array, L, c0, c1)
    x2.scatter_reverse(la.InsertMode.add)

    assert np.linalg.norm(x0.array - x1.array) == pytest.approx(0.0)
    assert np.linalg.norm(x0.array - x2.array) == pytest.approx(0.0, abs=1e-7)


@dtype_parametrize
@pytest.mark.parametrize("method", ["degree", "metadata"])
def test_mixed_quadrature(dtype, method):
    xtype = dtype(0).real.dtype
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, dtype=xtype)

    V = functionspace(mesh, ("Lagrange", 1))
    u = Function(V, dtype=dtype)
    u.interpolate(lambda x: x[0])

    tol = 500 * np.finfo(dtype).eps
    num_cells_local = (
        mesh.topology.index_map(mesh.topology.dim).size_local
        + mesh.topology.index_map(mesh.topology.dim).num_ghosts
    )
    values = np.full(num_cells_local, 1, dtype=np.int32)
    left_cells = locate_entities(mesh, mesh.topology.dim, lambda x: x[0] <= 0.5 + tol)
    values[left_cells] = 2
    top_cells = locate_entities(mesh, mesh.topology.dim, lambda x: x[1] >= 0.5 - tol)
    values[top_cells] = 3
    ct = meshtags(mesh, mesh.topology.dim, np.arange(num_cells_local, dtype=np.int32), values)

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)

    if method == "degree":
        dx_1 = dx(subdomain_id=(1,), degree=1)
        dx_2 = dx(subdomain_id=(1, 2), degree=2)
        dx_3 = dx(subdomain_id=(2, 3), degree=3)
    elif method == "metadata":
        dx_1 = dx(subdomain_id=(1,), metadata={"quadrature_degree": 1})
        dx_2 = dx(subdomain_id=(1, 2), metadata={"quadrature_degree": 2})
        dx_3 = dx(subdomain_id=(2, 3), metadata={"quadrature_degree": 3})
    else:
        raise ValueError(f"Invalid method {method}")
    form_1 = u * dx_1
    form_2 = u * dx_2
    form_3 = u * dx_3
    summed_form = form_1 + form_2 + form_3

    compiled_forms = form([form_1, form_2, form_3], dtype=dtype)
    local_contributions = 0
    for compiled_form in compiled_forms:
        local_contributions += assemble_scalar(compiled_form)
    global_contribution = mesh.comm.allreduce(local_contributions, op=MPI.SUM)

    compiled_form = form(summed_form, dtype=dtype)
    local_sum = assemble_scalar(compiled_form)
    global_sum = mesh.comm.allreduce(local_sum, op=MPI.SUM)
    assert np.isclose(global_contribution, global_sum, rtol=tol, atol=tol)


def vertex_to_dof_map(V):
    """Create a map from the vertices of the mesh to the corresponding degree of freedom."""
    mesh = V.mesh
    num_vertices_per_cell = dolfinx.cpp.mesh.cell_num_entities(mesh.topology.cell_type, 0)

    dof_layout2 = np.empty((num_vertices_per_cell,), dtype=np.int32)
    for i in range(num_vertices_per_cell):
        var = V.dofmap.dof_layout.entity_dofs(0, i)
        assert len(var) == 1
        dof_layout2[i] = var[0]

    num_vertices = mesh.topology.index_map(0).size_local + mesh.topology.index_map(0).num_ghosts

    c_to_v = mesh.topology.connectivity(mesh.topology.dim, 0)
    assert (
        c_to_v.num_nodes == 0
        or (c_to_v.offsets[1:] - c_to_v.offsets[:-1] == c_to_v.offsets[1]).all()
    ), "Single cell type supported"

    vertex_to_dof_map = np.empty(num_vertices, dtype=np.int32)
    vertex_to_dof_map[c_to_v.array] = V.dofmap.list[:, dof_layout2].reshape(-1)
    return vertex_to_dof_map


@pytest.mark.parametrize(
    "cell_type",
    [
        mesh.CellType.interval,
        mesh.CellType.triangle,
        mesh.CellType.quadrilateral,
        mesh.CellType.tetrahedron,
        # mesh.CellType.pyramid,
        mesh.CellType.prism,
        mesh.CellType.hexahedron,
    ],
)
@pytest.mark.parametrize("ghost_mode", [mesh.GhostMode.none, mesh.GhostMode.shared_facet])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_vertex_integral_rank_0(cell_type, ghost_mode, dtype):
    comm = MPI.COMM_WORLD
    rdtype = np.real(dtype(0)).dtype

    msh = None
    cell_dim = mesh.cell_dim(cell_type)
    if cell_dim == 1:
        msh = mesh.create_unit_interval(comm, 4, dtype=rdtype, ghost_mode=ghost_mode)
    elif cell_dim == 2:
        msh = mesh.create_unit_square(
            comm, 4, 4, cell_type=cell_type, dtype=rdtype, ghost_mode=ghost_mode
        )
    elif cell_dim == 3:
        msh = mesh.create_unit_cube(
            comm, 4, 4, 4, cell_type=cell_type, dtype=rdtype, ghost_mode=ghost_mode
        )
    else:
        raise RuntimeError("Bad dimension")

    vertex_map = msh.topology.index_map(0)

    def check_vertex_integral_against_sum(form, vertices, weighted=False):
        """Weighting assumes the vertex integral to be weighted by a P1 function, each vertex value
        corresponding to its global index."""
        weights = vertex_map.local_to_global(vertices) if weighted else np.ones_like(vertices)
        expected_value_l = np.sum(msh.geometry.x[vertices, 0] * weights)
        value_l = fem.assemble_scalar(fem.form(form, dtype=dtype))
        assert expected_value_l == pytest.approx(value_l, abs=5e4 * np.finfo(rdtype).eps)

        expected_value = comm.allreduce(expected_value_l)
        value = comm.allreduce(value_l)
        assert expected_value == pytest.approx(value, abs=5e4 * np.finfo(rdtype).eps)

    num_vertices = vertex_map.size_local
    x = ufl.SpatialCoordinate(msh)

    # Full domain
    check_vertex_integral_against_sum(x[0] * ufl.dP, np.arange(num_vertices))

    # Split domain into left half of vertices (1) and right half of vertices (2)
    vertices_1 = mesh.locate_entities(msh, 0, lambda x: x[0] <= 0.5)
    vertices_1 = vertices_1[vertices_1 < num_vertices]
    vertices_2 = mesh.locate_entities(msh, 0, lambda x: x[0] > 0.5)
    vertices_2 = vertices_2[vertices_2 < num_vertices]

    tags = np.full(num_vertices, 1)
    tags[vertices_2] = 2
    vertices = np.arange(0, num_vertices, dtype=np.int32)
    meshtags = mesh.meshtags(msh, 0, vertices, tags)

    dP = ufl.Measure("dP", domain=msh, subdomain_data=meshtags)

    # Combinations of sub domains
    check_vertex_integral_against_sum(x[0] * dP(1), vertices_1)
    check_vertex_integral_against_sum(x[0] * dP(2), vertices_2)
    check_vertex_integral_against_sum(x[0] * (dP(1) + dP(2)), np.arange(num_vertices))

    V = fem.functionspace(msh, ("P", 1))
    u = fem.Function(V, dtype=dtype)
    vertex_to_dof = vertex_to_dof_map(V)
    vertices = np.arange(num_vertices + vertex_map.num_ghosts)
    u.x.array[vertex_to_dof[vertices]] = vertex_map.local_to_global(vertices)

    check_vertex_integral_against_sum(u * x[0] * ufl.dP, np.arange(num_vertices), True)
    check_vertex_integral_against_sum(u * x[0] * dP(1), vertices_1, True)
    check_vertex_integral_against_sum(u * x[0] * dP(2), vertices_2, True)
    check_vertex_integral_against_sum(u * x[0] * (dP(1) + dP(2)), np.arange(num_vertices), True)

    # Check custom packing
    if cell_type is mesh.CellType.prism:
        return

    msh.topology.create_entities(1)
    msh.topology.create_connectivity(cell_dim - 1, cell_dim)

    v_to_c = msh.topology.connectivity(0, cell_dim)
    c_to_v = msh.topology.connectivity(cell_dim, 0)

    cell_vertex_pairs = np.array([], dtype=np.int32)
    for v in range(num_vertices):
        c = v_to_c.links(v)[0]
        v_l = np.where(c_to_v.links(c) == v)[0]
        cell_vertex_pairs = np.append(cell_vertex_pairs, [c, *v_l])

    # a) With subdomain_data
    check_vertex_integral_against_sum(
        x[0] * ufl.dP(domain=msh, subdomain_data=[(1, cell_vertex_pairs)], subdomain_id=1),
        np.arange(num_vertices),
    )

    # b) With create_form
    vertices = np.arange(num_vertices)
    fem.compute_integration_domains(fem.IntegralType.exterior_facet, msh.topology, vertices)
    subdomains = {fem.IntegralType.exterior_facet: [(0, cell_vertex_pairs)]}

    compiled_form = fem.compile_form(
        comm, x[0] * ufl.dP, form_compiler_options={"scalar_type": dtype}
    )
    form = fem.create_form(compiled_form, [], msh, subdomains, {}, {}, [])
    expected_value_l = np.sum(msh.geometry.x[vertices, 0])
    value_l = fem.assemble_scalar(form)
    assert expected_value_l == pytest.approx(value_l, abs=5e4 * np.finfo(rdtype).eps)


@pytest.mark.parametrize(
    "cell_type",
    [
        mesh.CellType.interval,
        mesh.CellType.triangle,
        mesh.CellType.quadrilateral,
        mesh.CellType.tetrahedron,
        # mesh.CellType.pyramid,
        mesh.CellType.prism,
        mesh.CellType.hexahedron,
    ],
)
@pytest.mark.parametrize("ghost_mode", [mesh.GhostMode.none, mesh.GhostMode.shared_facet])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_vertex_integral_rank_1(cell_type, ghost_mode, dtype):
    comm = MPI.COMM_WORLD
    rdtype = np.real(dtype(0)).dtype

    msh = None
    cell_dim = mesh.cell_dim(cell_type)
    if cell_dim == 1:
        msh = mesh.create_unit_interval(comm, 4, ghost_mode=ghost_mode, dtype=rdtype)
    elif cell_dim == 2:
        msh = mesh.create_unit_square(
            comm, 4, 4, cell_type=cell_type, ghost_mode=ghost_mode, dtype=rdtype
        )
    elif cell_dim == 3:
        msh = mesh.create_unit_cube(
            comm, 4, 4, 4, cell_type=cell_type, ghost_mode=ghost_mode, dtype=rdtype
        )
    else:
        raise RuntimeError("Bad dimension")

    vertex_map = msh.topology.index_map(0)
    num_vertices = vertex_map.size_local

    def check_vertex_integral_against_sum(form, vertices, weighted=False):
        """Weighting assumes the vertex integral to be weighted by a P1 function, each vertex value
        corresponding to its global index."""
        weights = vertex_map.local_to_global(vertices) if weighted else np.ones_like(vertices)
        expected_value_l = np.zeros(num_vertices, dtype=rdtype)
        expected_value_l[vertices] = msh.geometry.x[vertices, 0] * weights
        value_l = fem.assemble_vector(fem.form(form, dtype=dtype))
        equal_l = np.allclose(
            expected_value_l, np.real(value_l.array[:num_vertices]), atol=1e3 * np.finfo(rdtype).eps
        )
        assert equal_l
        assert comm.allreduce(equal_l, MPI.BAND)

    x = ufl.SpatialCoordinate(msh)
    V = fem.functionspace(msh, ("P", 1))
    v = ufl.conj(ufl.TestFunction(V))

    # Full domain
    check_vertex_integral_against_sum(x[0] * v * ufl.dP, np.arange(num_vertices))

    # Split domain into left half of vertices (1) and right half of vertices (2)
    vertices_1 = mesh.locate_entities(msh, 0, lambda x: x[0] <= 0.5)
    vertices_1 = vertices_1[vertices_1 < num_vertices]
    vertices_2 = mesh.locate_entities(msh, 0, lambda x: x[0] > 0.5)
    vertices_2 = vertices_2[vertices_2 < num_vertices]

    tags = np.full(num_vertices, 1)
    tags[vertices_2] = 2
    vertices = np.arange(0, num_vertices, dtype=np.int32)
    meshtags = mesh.meshtags(msh, 0, vertices, tags)

    dP = ufl.Measure("dP", domain=msh, subdomain_data=meshtags)

    check_vertex_integral_against_sum(x[0] * v * dP(1), vertices_1)
    check_vertex_integral_against_sum(x[0] * v * dP(2), vertices_2)
    check_vertex_integral_against_sum(x[0] * v * (dP(1) + dP(2)), np.arange(num_vertices))

    V = fem.functionspace(msh, ("P", 1))
    u = fem.Function(V, dtype=dtype)
    u.x.array[:] = vertex_map.local_to_global(np.arange(num_vertices + vertex_map.num_ghosts))
    vertex_to_dof = vertex_to_dof_map(V)
    vertices = np.arange(num_vertices + vertex_map.num_ghosts)
    u.x.array[vertex_to_dof[vertices]] = vertex_map.local_to_global(vertices)

    check_vertex_integral_against_sum(u * x[0] * v * ufl.dP, np.arange(num_vertices), True)
    check_vertex_integral_against_sum(u * x[0] * v * dP(1), vertices_1, True)
    check_vertex_integral_against_sum(u * x[0] * v * dP(2), vertices_2, True)
    check_vertex_integral_against_sum(u * x[0] * v * (dP(1) + dP(2)), np.arange(num_vertices), True)

    # Check custom packing
    if cell_type is mesh.CellType.prism:
        return

    msh.topology.create_entities(1)
    msh.topology.create_connectivity(cell_dim - 1, cell_dim)

    v_to_c = msh.topology.connectivity(0, cell_dim)
    c_to_v = msh.topology.connectivity(cell_dim, 0)

    cell_vertex_pairs = np.array([], dtype=np.int32)
    for v in range(num_vertices):
        c = v_to_c.links(v)[0]
        v_l = np.where(c_to_v.links(c) == v)[0]
        cell_vertex_pairs = np.append(cell_vertex_pairs, [c, *v_l])

    # a) With subdomain_data
    v = ufl.conj(ufl.TestFunction(V))
    check_vertex_integral_against_sum(
        x[0] * v * ufl.dP(domain=msh, subdomain_data=[(1, cell_vertex_pairs)], subdomain_id=1),
        np.arange(num_vertices),
    )

    # b) With create_form
    vertices = np.arange(num_vertices)
    fem.compute_integration_domains(fem.IntegralType.exterior_facet, msh.topology, vertices)
    subdomains = {fem.IntegralType.exterior_facet: [(0, cell_vertex_pairs)]}

    compiled_form = fem.compile_form(
        comm, x[0] * v * ufl.dP, form_compiler_options={"scalar_type": dtype}
    )
    form = fem.create_form(compiled_form, [V], msh, subdomains, {}, {}, [])
    expected_value_l = np.sum(msh.geometry.x[vertices, 0])
    expected_value_l = np.zeros(num_vertices, dtype=rdtype)
    expected_value_l[vertices] = msh.geometry.x[vertices, 0]
    value_l = fem.assemble_vector(form)
    assert expected_value_l == pytest.approx(
        value_l.array[: expected_value_l.size], abs=5e4 * np.finfo(rdtype).eps
    )
