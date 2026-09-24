# Copyright (C) 2026 Paul T. Kühner
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest

from dolfinx import mesh
from dolfinx.common import index_map


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


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mark_equidistribution_empty(dtype: np.dtype) -> None:
    """An index map with no entries marks nothing."""
    im = index_map(MPI.COMM_WORLD, 0)
    marked = mesh.mark_equidistribution(np.zeros(0, dtype=dtype), im, 0.5)
    assert marked.size == 0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mark_equidistribution_ignores_ghosts(dtype: np.dtype) -> None:
    """The mean square is reduced over owned entries only.

    Rank 0 owns one entry per process, holding the values
    ``0, ..., size - 1``. Every other rank owns nothing and ghosts entry
    ``rank``, which it sets to ``ghost_value``. That value is chosen large
    enough that a reduction wrongly including ghosts would push the
    threshold above every owned value; the guard below asserts this, since
    a ghost that is merely larger than the owned values leaves the marked
    set unchanged and the test vacuous.
    """
    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size

    local_size = size if rank == 0 else 0
    ghosts = np.zeros(0, dtype=np.int64) if rank == 0 else np.array([rank], dtype=np.int64)
    owners = np.zeros(0, dtype=np.int32) if rank == 0 else np.array([0], dtype=np.int32)
    im = index_map(comm, local_size, (ghosts, owners), tag=0)

    ghost_value = 100 * size
    v = np.empty(im.size_local + im.num_ghosts, dtype=dtype)
    if rank == 0:
        v[:] = np.arange(size, dtype=dtype)
    else:
        v[0] = ghost_value

    theta = 0.5
    marked = mesh.mark_equidistribution(v, im, theta)

    # Sum over the owned entries only, i.e. 0 + 1 + ... + (size - 1)
    norm = size * (size - 1) / 2
    threshold = theta**2 * norm / im.size_global
    assert np.array_equal(marked, np.flatnonzero(v > threshold))

    # Guard: on rank 0 the marked set must differ from what a reduction
    # over owned + ghost entries would give, or the test proves nothing.
    if rank == 0 and size > 1:
        norm_with_ghosts = norm + (size - 1) * ghost_value
        threshold_with_ghosts = theta**2 * norm_with_ghosts / im.size_global
        assert not np.array_equal(marked, np.flatnonzero(v > threshold_with_ghosts))
