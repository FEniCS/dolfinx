# Copyright (C) 2026 Paul T. Kühner
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest

import dolfinx
from dolfinx import la, mesh


@pytest.mark.parametrize("theta", [0.2, 0.4, 0.6, 0.8])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "ghost_mode", [dolfinx.mesh.GhostMode.none, dolfinx.mesh.GhostMode.shared_facet]
)
def test_mark_maximum(theta: float, dtype: np.dtype, ghost_mode: dolfinx.mesh.GhostMode) -> None:
    msh = mesh.create_unit_square(
        comm := MPI.COMM_WORLD, n := 10, n, dtype=dtype, ghost_mode=ghost_mode
    )
    tdim = msh.topology.dim

    im_c = msh.topology.index_map(tdim)
    marker = la.vector(im_c, dtype=dtype)
    marker.array[: marker.index_map.size_local] = np.random.default_rng(0).random(
        marker.index_map.size_local
    )
    marker.scatter_forward()

    marked_cells = mesh.mark_maximum(marker, theta)

    threshold = theta * comm.allreduce(np.max(marker.array), MPI.MAX)
    assert np.allclose(
        marked_cells,
        np.argwhere(marker.array > threshold).flatten(),
    )

    msh.topology.create_entities(1)
    marked_edges = mesh.compute_incident_entities(msh.topology, marked_cells, tdim, 1)
    mesh.refine(msh, marked_edges)
