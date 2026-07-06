# Copyright (C) 2026 Jack S. Hale, Chris Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for SuperLU_DIST."""

from mpi4py import MPI

import numpy as np
import pytest

import dolfinx
from dolfinx.common import IndexMap
from dolfinx.cpp.la import SparsityPattern
from dolfinx.fem import (
    Function,
    apply_lifting,
    assemble_matrix,
    assemble_scalar,
    assemble_vector,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.la import InsertMode, matrix_csr, vector
from dolfinx.mesh import create_unit_square, exterior_facet_indices
from ufl import SpatialCoordinate, TestFunction, TrialFunction, as_vector, div, dx, grad, inner


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128])
@pytest.mark.skipif(not dolfinx.has_superlu_dist, reason="No SuperLU_DIST")
def test_superlu_solver(dtype):
    """Manufactured Poisson and screened problem with exact solution u = x[1]**3.

    Test includes various checks that SuperLU_DIST Factor option works correctly
    for efficient re-use of permutations and factors.
    """
    from dolfinx.la.superlu_dist import superlu_dist_matrix, superlu_dist_solver

    mesh_dtype = dtype().real.dtype
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5, dtype=mesh_dtype)
    V = functionspace(mesh, ("Lagrange", 4))
    u, v = TrialFunction(V), TestFunction(V)

    a = inner(grad(u), grad(v)) * dx
    a = form(a, dtype=dtype)

    # Exact solution
    def u_ex(x):
        return x[1] ** 3

    x = SpatialCoordinate(mesh)
    f = -div(grad(u_ex(x)))
    L = inner(f, v) * dx
    L = form(L, dtype=dtype)

    u_bc = Function(V, dtype=dtype)
    u_bc.interpolate(u_ex)

    # Create Dirichlet boundary condition
    facetdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(facetdim, mesh.topology.dim)
    bndry_facets = exterior_facet_indices(mesh.topology)
    bdofs = locate_dofs_topological(V, facetdim, bndry_facets)
    bc = dirichletbc(u_bc, bdofs)

    b = assemble_vector(L)
    apply_lifting(b.array, [a], bcs=[[bc]])
    b.scatter_reverse(InsertMode.add)
    bc.set(b.array)

    a = form(a, dtype=dtype)
    A = assemble_matrix(a, bcs=[bc])
    A.scatter_reverse()

    def check_error(u_ex, uh):
        M = (u_ex(x) - uh) ** 2 * dx
        M = form(M, dtype=dtype)
        error = mesh.comm.allreduce(assemble_scalar(M), op=MPI.SUM)
        eps = np.sqrt(np.finfo(dtype).eps)
        assert np.isclose(error, 0.0, atol=eps)

    def solve_and_check(solver, b):
        uh = Function(V, dtype=dtype)
        error_code = solver.solve(b, uh.x)
        assert error_code == 0
        uh.x.scatter_forward()
        return uh

    # Standard solve
    A_superlu = superlu_dist_matrix(A)
    solver_1 = superlu_dist_solver(A_superlu)
    solver_1.set_option("SymmetricMode", "YES")
    uh_1 = solve_and_check(solver_1, b)
    check_error(u_ex, uh_1)

    # Check second solve re-using permutations from previous solve
    solver_1.set_option("Fact", "FACTORED")
    solve_and_check(solver_1, b)
    uh_2 = solve_and_check(solver_1, b)
    check_error(u_ex, uh_2)

    # Check can do same solve with MatrixCSR A again.
    A_superlu_2 = superlu_dist_matrix(A)
    solver_2 = superlu_dist_solver(A_superlu_2)
    solver_2.set_option("SymmetricMode", "YES")
    uh_3 = solve_and_check(solver_2, b)
    check_error(u_ex, uh_3)

    # Assemble a new operator and re-use row permutation from solve with old
    # operator.
    gamma = 1e-1
    a_2 = (inner(grad(u), grad(v)) + gamma * inner(u, v)) * dx
    a_2 = form(a_2, dtype=dtype)

    f_2 = -div(grad(u_ex(x))) + gamma * u_ex(x)
    L_2 = inner(f_2, v) * dx
    L_2 = form(L_2, dtype=dtype)

    b_2 = assemble_vector(L_2)
    apply_lifting(b_2.array, [a_2], bcs=[[bc]])
    b_2.scatter_reverse(InsertMode.add)
    bc.set(b_2.array)

    A_3 = assemble_matrix(a_2, bcs=[bc])
    A_3.scatter_reverse()

    A_superlu_3 = superlu_dist_matrix(A_3)
    solver_2.set_A(A_superlu_3, "SamePattern")
    uh_2 = solve_and_check(solver_2, b_2)
    check_error(u_ex, uh_2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128])
@pytest.mark.skipif(not dolfinx.has_superlu_dist, reason="No SuperLU_DIST")
def test_superlu_solver_blocked(dtype):
    """Vector Poisson problem on a vector Lagrange space (block size 2)."""
    from dolfinx.la.superlu_dist import superlu_dist_matrix, superlu_dist_solver

    mesh_dtype = dtype().real.dtype
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5, dtype=mesh_dtype)
    V = functionspace(mesh, ("Lagrange", 3, (2,)))
    u, v = TrialFunction(V), TestFunction(V)

    a = form(inner(grad(u), grad(v)) * dx, dtype=dtype)

    def u_ex(x):
        return np.vstack((x[1] ** 3, x[0] ** 3))

    x = SpatialCoordinate(mesh)
    u_ex_ufl = as_vector((x[1] ** 3, x[0] ** 3))
    f = -div(grad(u_ex_ufl))
    L = form(inner(f, v) * dx, dtype=dtype)

    u_bc = Function(V, dtype=dtype)
    u_bc.interpolate(u_ex)

    facetdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(facetdim, mesh.topology.dim)
    bndry_facets = exterior_facet_indices(mesh.topology)
    bdofs = locate_dofs_topological(V, facetdim, bndry_facets)
    bc = dirichletbc(u_bc, bdofs)

    b = assemble_vector(L)
    apply_lifting(b.array, [a], bcs=[[bc]])
    b.scatter_reverse(InsertMode.add)
    bc.set(b.array)

    A = assemble_matrix(a, bcs=[bc])
    A.scatter_reverse()
    assert A.block_size == [2, 2]

    A_superlu = superlu_dist_matrix(A)
    solver = superlu_dist_solver(A_superlu)
    solver.set_option("SymmetricMode", "YES")
    uh = Function(V, dtype=dtype)
    error_code = solver.solve(b, uh.x)
    assert error_code == 0
    uh.x.scatter_forward()

    M = form(inner(u_ex_ufl - uh, u_ex_ufl - uh) * dx, dtype=dtype)
    error = mesh.comm.allreduce(assemble_scalar(M), op=MPI.SUM)
    eps = np.sqrt(np.finfo(dtype).eps)
    assert np.isclose(error, 0.0, atol=eps)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128])
@pytest.mark.skipif(not dolfinx.has_superlu_dist, reason="No SuperLU_DIST")
@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="Hand-built single-rank matrix")
def test_superlu_solver_asymmetric_blocks(dtype):
    """Hand-built MatrixCSR with bs[0] = 2 and bs[1] = 3 and final size 6 x 6."""
    from dolfinx.la.superlu_dist import superlu_dist_matrix, superlu_dist_solver

    bs0, bs1 = 2, 3
    n_row_blocks, n_col_blocks = 3, 2

    im_row = IndexMap(MPI.COMM_WORLD, n_row_blocks)
    im_col = IndexMap(MPI.COMM_WORLD, n_col_blocks)
    sp = SparsityPattern(MPI.COMM_WORLD, [im_row, im_col], [bs0, bs1])
    for i in range(n_row_blocks):
        for j in range(n_col_blocks):
            sp.insert(i, j)
    sp.finalize()

    A = matrix_csr(sp, dtype=dtype)
    assert A.block_size == [bs0, bs1]

    rng = np.random.default_rng(0)
    A_dense = (np.eye(6) * 10.0 + rng.standard_normal((6, 6))).astype(dtype)

    for i in range(n_row_blocks):
        for j in range(n_col_blocks):
            block_idx = i * n_col_blocks + j
            for i0 in range(bs0):
                for i1 in range(bs1):
                    A.data[block_idx * bs0 * bs1 + i0 * bs1 + i1] = A_dense[
                        i * bs0 + i0, j * bs1 + i1
                    ]

    b_np = rng.standard_normal(6).astype(dtype)
    x_expected = np.linalg.solve(A_dense, b_np)

    b = vector(im_row, bs=bs0, dtype=dtype)
    b.array[:] = b_np
    u = vector(im_col, bs=bs1, dtype=dtype)

    A_superlu = superlu_dist_matrix(A)
    solver = superlu_dist_solver(A_superlu)
    error_code = solver.solve(b, u)
    assert error_code == 0
    u.scatter_forward()

    assert np.allclose(u.array, x_expected)
