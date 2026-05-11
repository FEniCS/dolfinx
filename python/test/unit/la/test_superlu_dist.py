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
from dolfinx.la import InsertMode
from dolfinx.mesh import create_unit_square, exterior_facet_indices
from ufl import SpatialCoordinate, TestFunction, TrialFunction, div, dx, grad, inner


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
    solver_2.set_A(A_superlu_3)
    solver_2.set_option("Fact", "SamePattern")
    uh_2 = solve_and_check(solver_2, b_2)
    check_error(u_ex, uh_2)
