# Copyright (C) 2026 Paul T. Kühner
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest

import ufl
from dolfinx import default_real_type, fem, la, mesh


@pytest.mark.parametrize("theta", [0.2, 0.4, 0.6, 0.8])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mark_maximum(theta: float, dtype: np.dtype) -> None:
    comm = MPI.COMM_WORLD
    n = 10
    msh = mesh.create_unit_square(comm, n, n, dtype=dtype)

    tdim = msh.topology.dim
    cell_count = msh.topology.index_map(tdim).size_local
    indicators = np.random.default_rng(0).random(cell_count, dtype=dtype)

    marked_cells = mesh.mark_maximum(comm, indicators, theta)

    assert np.allclose(
        marked_cells,
        np.argwhere(indicators > theta * comm.allreduce(np.max(indicators), MPI.MAX)).flatten(),
    )

    msh.topology.create_entities(1)
    marked_edges = mesh.compute_incident_entities(msh.topology, marked_cells, tdim, 1)
    mesh.refine(msh, marked_edges)


@pytest.mark.parametrize("theta", [0.2, 0.4, 0.6, 0.8])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mark_equidistribution(theta: float, dtype: np.dtype) -> None:
    comm = MPI.COMM_WORLD
    n = 10
    msh = mesh.create_unit_square(comm, n, n, dtype=dtype)

    tdim = msh.topology.dim
    cell_count = msh.topology.index_map(tdim).size_local
    indicators = np.random.default_rng(0).random(cell_count, dtype=dtype)

    marked_cells = mesh.mark_equidistribution(comm, indicators, theta)

    norm = np.sqrt(comm.allreduce(np.sum(indicators**2), MPI.SUM))
    count = comm.allreduce(indicators.size)
    assert np.allclose(
        marked_cells,
        np.argwhere(indicators > theta * norm / np.sqrt(count)).flatten(),
    )

    msh.topology.create_entities(1)
    marked_edges = mesh.compute_incident_entities(msh.topology, marked_cells, tdim, 1)
    mesh.refine(msh, marked_edges)

    assert np.all(marked_cells == mesh.mark_equidistribution_squared(comm, indicators**2, theta))


@pytest.mark.parametrize("theta", [0.2, 0.4, 0.6, 0.8])
def test_mark_poisson_residual_estimator(theta: float, cg_solver) -> None:
    r"""Solve a Poisson problem, compute the standard residual a posteriori
    error estimator, and use it to drive equidistribution marking and refinement.
    """
    comm = MPI.COMM_WORLD
    dtype = default_real_type
    msh = mesh.create_unit_square(comm, 8, 8, dtype=default_real_type)
    tdim = msh.topology.dim

    # Solve Poisson with Gaussian bump.
    V = fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    f = ufl.exp(-((x[0] - 0.25) ** 2 + (x[1] - 0.25) ** 2) / 0.02)
    a = fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx, dtype=dtype)
    L = fem.form(ufl.inner(f, v) * ufl.dx, dtype=dtype)

    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    bdofs = fem.locate_dofs_topological(V, fdim, mesh.exterior_facet_indices(msh.topology))
    bc = fem.dirichletbc(dtype(0), bdofs, V)

    A = fem.assemble_matrix(a, bcs=[bc])
    A.scatter_reverse()
    b = fem.assemble_vector(L)
    fem.apply_lifting(b.array, [a], bcs=[[bc]])
    b.scatter_reverse(la.InsertMode.add)
    bc.set(b.array)

    uh = fem.Function(V, dtype=dtype)
    cg_solver(msh.comm, A, b, uh.x)
    uh.x.scatter_forward()

    # Assemble squared cell indicators eta_K^2 into a DG0 vector.
    V0 = fem.functionspace(msh, ("DG", 0))
    eta_T = ufl.TestFunction(V0)
    h = ufl.CellDiameter(msh)
    n = ufl.FacetNormal(msh)
    j = ufl.jump(ufl.grad(uh), n)
    eta_form = fem.form(
        h**2 * ufl.inner(f, f) * eta_T * ufl.dx
        + ufl.avg(h) * ufl.inner(j, j) * ufl.avg(eta_T) * ufl.dS,
        dtype=dtype,
    )
    eta_vec = fem.assemble_vector(eta_form)
    eta_vec.scatter_reverse(la.InsertMode.add)

    indicators_squared = eta_vec.array[: V0.dofmap.index_map.size_local]
    assert np.all(indicators_squared >= 0)

    # This assumes an identity mapping from V0 to cells... True?
    marked_cells = mesh.mark_equidistribution_squared(comm, indicators_squared, theta)

    msh.topology.create_entities(1)
    marked_edges = mesh.compute_incident_entities(msh.topology, marked_cells, tdim, 1)
    mesh.refine(msh, marked_edges)
