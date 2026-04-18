# Copyright (C) 2026 Paul T. Kühner
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest

from dolfinx import mesh


@pytest.mark.parametrize("theta", np.linspace(0, 1, num=5, endpoint=True))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mark_maximum(theta: float, dtype: np.dtype) -> None:
    msh = mesh.create_unit_square(comm := MPI.COMM_WORLD, n := 10, n, dtype=dtype)

    tdim = msh.topology.dim
    cell_count = (cell_im := msh.topology.index_map(tdim)).size_local + cell_im.num_ghosts
    marker = np.random.default_rng(0).random(cell_count)

    marked_cells = mesh.mark_maximum(marker, theta, comm)

    assert np.allclose(
        marked_cells,
        np.argwhere(marker >= theta * comm.allreduce(np.max(marker), MPI.MAX)).flatten(),
    )

    msh.topology.create_entities(1)
    marked_edges = mesh.compute_incident_entities(msh.topology, marked_cells, tdim, 1)
    mesh.refine(msh, marked_edges)
