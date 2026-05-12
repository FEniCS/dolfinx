# Copyright (C) 2026 Paul T. Kühner
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest

from dolfinx import mesh


@pytest.mark.parametrize("theta", [0.2, 0.4, 0.6, 0.8])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mark_maximum(theta: float, dtype: np.dtype) -> None:
    comm = MPI.COMM_WORLD
    n = 10
    msh = mesh.create_unit_square(comm, n, n, dtype=dtype)

    tdim = msh.topology.dim
    cell_count = (cell_im := msh.topology.index_map(tdim)).size_local + cell_im.num_ghosts
    marker = np.random.default_rng(0).random(cell_count, dtype=dtype)

    marked_cells = mesh.mark_maximum(comm, marker, theta)

    assert np.allclose(
        marked_cells,
        np.argwhere(marker > theta * comm.allreduce(np.max(marker), MPI.MAX)).flatten(),
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
    cell_count = (cell_im := msh.topology.index_map(tdim)).size_local + cell_im.num_ghosts
    marker = np.random.default_rng(0).random(cell_count, dtype=dtype)

    marked_cells = mesh.mark_equidistribution(comm, marker, theta)

    norm = np.sqrt(comm.allreduce(np.sum(marker**2), MPI.SUM))
    count = comm.allreduce(marker.size)
    assert np.allclose(
        marked_cells,
        np.argwhere(marker > theta * norm / np.sqrt(count)).flatten(),
    )

    msh.topology.create_entities(1)
    marked_edges = mesh.compute_incident_entities(msh.topology, marked_cells, tdim, 1)
    mesh.refine(msh, marked_edges)

    assert np.all(marked_cells == mesh.mark_equidistribution_squared(comm, marker**2, theta))
