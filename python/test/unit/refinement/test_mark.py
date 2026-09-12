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
@pytest.mark.parametrize("ghost_mode", [mesh.GhostMode.none, mesh.GhostMode.shared_facet])
def test_mark_maximum(theta: float, dtype: np.dtype, ghost_mode: mesh.GhostMode) -> None:
    msh = mesh.create_unit_square(
        comm := MPI.COMM_WORLD, n := 10, n, dtype=dtype, ghost_mode=ghost_mode
    )
    tdim = msh.topology.dim

    im_c = msh.topology.index_map(tdim)
    marker = np.random.default_rng(0).random(im_c.size_local + im_c.num_ghosts)

    marked_cells = mesh.mark_maximum(marker, im_c, theta)

    threshold = theta * comm.allreduce(np.max(marker), MPI.MAX)
    assert np.allclose(marked_cells, np.argwhere(marker > threshold).flatten())

    msh.topology.create_entities(1)
    marked_edges = mesh.compute_incident_entities(msh.topology, marked_cells, tdim, 1)
    mesh.refine(msh, marked_edges)


@pytest.mark.parametrize("theta", [0.2, 0.4, 0.6, 0.8])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("ghost_mode", [mesh.GhostMode.none, mesh.GhostMode.shared_facet])
def test_mark_equidistribution(theta: float, dtype: np.dtype, ghost_mode: mesh.GhostMode) -> None:
    msh = mesh.create_unit_square(
        comm := MPI.COMM_WORLD, n := 10, n, dtype=dtype, ghost_mode=ghost_mode
    )
    tdim = msh.topology.dim

    im_c = msh.topology.index_map(tdim)
    marker = np.random.default_rng(0).random(im_c.size_local + im_c.num_ghosts, dtype=dtype)

    marked_cells = mesh.mark_equidistribution(marker, im_c, theta)

    # Note: run equidistribution check on squared inequality
    norm = comm.allreduce(np.sum(marker[: im_c.size_local]))
    threshold = theta**2 * norm / im_c.size_global
    assert np.allclose(marked_cells, np.argwhere(marker > threshold).flatten())

    msh.topology.create_entities(1)
    marked_edges = mesh.compute_incident_entities(msh.topology, marked_cells, tdim, 1)
    mesh.refine(msh, marked_edges)
