# Copyright (C) 2016 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest

from dolfinx.mesh import (
    GhostMode,
    compute_midpoints,
    create_unit_cube,
    create_unit_interval,
    create_unit_square,
)


def test_ghost_facet_1d():
    N = 40
    mesh = create_unit_interval(MPI.COMM_WORLD, N, ghost_mode=GhostMode.shared_facet)
    assert mesh.topology.index_map(0).size_global == N + 1
    assert mesh.topology.index_map(1).size_global == N


def test_ghost_2d():
    N = 8
    num_cells = N * N * 2
    mesh = create_unit_square(MPI.COMM_WORLD, N, N, ghost_mode=GhostMode.shared_facet)
    if mesh.comm.size > 1:
        map = mesh.topology.index_map(2)
        num_cells_local = map.size_local + map.num_ghosts
        assert mesh.comm.allreduce(num_cells_local, op=MPI.SUM) > num_cells
    assert mesh.topology.index_map(0).size_global == 81
    assert mesh.topology.index_map(2).size_global == num_cells


def test_ghost_3d():
    N = 2
    num_cells = N * N * N * 6
    mesh = create_unit_cube(MPI.COMM_WORLD, N, N, N, ghost_mode=GhostMode.shared_facet)
    if mesh.comm.size > 1:
        map = mesh.topology.index_map(3)
        num_cells_local = map.size_local + map.num_ghosts
        assert mesh.comm.allreduce(num_cells_local, op=MPI.SUM) > num_cells
    assert mesh.topology.index_map(0).size_global == 27
    assert mesh.topology.index_map(3).size_global == num_cells


@pytest.mark.parametrize(
    "mode",
    [
        GhostMode.none,
        GhostMode.shared_facet,
    ],
)
def test_ghost_connectivities(mode):
    # Ghosted mesh
    meshG = create_unit_square(MPI.COMM_WORLD, 4, 4, ghost_mode=mode)
    meshG.topology.create_connectivity(1, 2)

    # Reference mesh, not ghosted, not parallel
    meshR = create_unit_square(MPI.COMM_SELF, 4, 4, ghost_mode=GhostMode.none)
    meshR.topology.create_connectivity(1, 2)
    tdim = meshR.topology.dim

    # Create reference map from facet midpoint to cell midpoint
    topology = meshR.topology
    map_c = topology.index_map(tdim)
    num_cells = map_c.size_local + map_c.num_ghosts
    map_f = topology.index_map(tdim - 1)
    num_facets = map_f.size_local + map_f.num_ghosts

    meshR.topology.create_connectivity(tdim - 1, tdim)
    facet_mp = compute_midpoints(meshR, tdim - 1, np.arange(num_facets))
    meshR.topology.create_connectivity(tdim, tdim)
    cell_mp = compute_midpoints(meshR, tdim, np.arange(num_cells))
    reference = {tuple(row): [] for row in facet_mp}
    for i in range(num_facets):
        for cidx in meshR.topology.connectivity(1, 2).links(i):
            reference[tuple(facet_mp[i])].append(cell_mp[cidx].tolist())

    # Loop through ghosted mesh and check connectivities
    tdim = meshG.topology.dim

    topology = meshG.topology
    map_c = topology.index_map(tdim)
    num_cells = map_c.size_local + map_c.num_ghosts
    map_f = topology.index_map(tdim - 1)
    num_facets = map_f.size_local + map_f.num_ghosts

    num_facets_ghost = map_f.num_ghosts
    allowable_cell_indices = range(num_cells)
    meshG.topology.create_connectivity(tdim - 1, tdim)
    facet_mp = compute_midpoints(meshG, tdim - 1, np.arange(num_facets))
    meshG.topology.create_connectivity(tdim, tdim)
    cell_mp = compute_midpoints(meshG, tdim, np.arange(num_cells))
    for i in range(num_facets_ghost):
        assert tuple(facet_mp[i]) in reference
        for cidx in meshG.topology.connectivity(1, 2).links(i):
            assert cidx in allowable_cell_indices
            assert cell_mp[cidx].tolist() in reference[tuple(facet_mp[i])]
