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
    """Manufactured Poisson problem with exact solution u = x[1]**3."""
    from dolfinx.la.superlu_dist import superlu_dist_solver

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

    solver = superlu_dist_solver(A)
    solver.set_option("SymmetricMode", "YES")

    uh = Function(V, dtype=dtype)
    error_code = solver.solve(b, uh.x)
    assert error_code == 0
    uh.x.scatter_forward()

    M = (u_ex(x) - uh) ** 2 * dx
    M = form(M, dtype=dtype)
    error = mesh.comm.allreduce(assemble_scalar(M), op=MPI.SUM)
    eps = np.sqrt(np.finfo(dtype).eps)
    assert np.isclose(error, 0.0, atol=eps)

    solver.set_option("Fact", "FACTORED")
    
    uh = Function(V, dtype=dtype)
    error_code = solver.solve(b, uh.x)
    assert error_code == 0
    uh.x.scatter_forward()

    M = (u_ex(x) - uh) ** 2 * dx
    M = form(M, dtype=dtype)
    error = mesh.comm.allreduce(assemble_scalar(M), op=MPI.SUM)
    eps = np.sqrt(np.finfo(dtype).eps)
    assert np.isclose(error, 0.0, atol=eps)
