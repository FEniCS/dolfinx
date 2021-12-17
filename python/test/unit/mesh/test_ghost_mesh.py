# Copyright (C) 2016 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy
import pytest

from dolfinx.mesh import (CellType, GhostMode, compute_midpoints, create_unit_cube,
                          create_unit_interval, create_unit_square,
                          compute_interface_facets, update_ghosts)
from dolfinx.graph import create_adjacencylist
from mpi4py import MPI


@pytest.mark.xfail(reason="Shared vertex currently disabled")
def test_ghost_vertex_1d():
    mesh = create_unit_interval(MPI.COMM_WORLD, 20, ghost_mode=GhostMode.shared_vertex)
    assert mesh.topology.index_map(0).size_global == 21
    assert mesh.topology.index_map(1).size_global == 20


def test_ghost_facet_1d():
    N = 40
    mesh = create_unit_interval(MPI.COMM_WORLD, N, ghost_mode=GhostMode.shared_facet)
    assert mesh.topology.index_map(0).size_global == N + 1
    assert mesh.topology.index_map(1).size_global == N


@pytest.mark.parametrize("mode",
                         [GhostMode.shared_facet,
                          pytest.param(GhostMode.shared_vertex,
                                       marks=pytest.mark.xfail(reason="Shared vertex currently disabled"))])
def test_ghost_2d(mode):
    N = 8
    num_cells = N * N * 2
    mesh = create_unit_square(MPI.COMM_WORLD, N, N, ghost_mode=mode)
    if mesh.comm.size > 1:
        map = mesh.topology.index_map(2)
        num_cells_local = map.size_local + map.num_ghosts
        assert mesh.comm.allreduce(num_cells_local, op=MPI.SUM) > num_cells
    assert mesh.topology.index_map(0).size_global == 81
    assert mesh.topology.index_map(2).size_global == num_cells


@pytest.mark.parametrize("mode", [GhostMode.shared_facet])
def test_ghost_3d(mode):
    N = 2
    num_cells = N * N * N * 6
    mesh = create_unit_cube(MPI.COMM_WORLD, N, N, N, ghost_mode=mode)
    if mesh.comm.size > 1:
        map = mesh.topology.index_map(3)
        num_cells_local = map.size_local + map.num_ghosts
        assert mesh.comm.allreduce(num_cells_local, op=MPI.SUM) > num_cells
    assert mesh.topology.index_map(0).size_global == 27
    assert mesh.topology.index_map(3).size_global == num_cells


@pytest.mark.parametrize("mode",
                         [GhostMode.none, GhostMode.shared_facet,
                          pytest.param(GhostMode.shared_vertex,
                                       marks=pytest.mark.xfail(reason="Shared vertex currently disabled"))])
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

    reference = {}
    facet_mp = compute_midpoints(meshR, tdim - 1, range(num_facets))
    cell_mp = compute_midpoints(meshR, tdim, range(num_cells))
    reference = dict.fromkeys([tuple(row) for row in facet_mp], [])
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
    facet_mp = compute_midpoints(meshG, tdim - 1, range(num_facets))
    cell_mp = compute_midpoints(meshG, tdim, range(num_cells))
    for i in range(num_facets_ghost):
        assert tuple(facet_mp[i]) in reference
        for cidx in meshG.topology.connectivity(1, 2).links(i):
            assert cidx in allowable_cell_indices
            assert cell_mp[cidx].tolist() in reference[tuple(facet_mp[i])]


@pytest.mark.parametrize("cell_type", [CellType.tetrahedron, CellType.hexahedron])
def test_ghost_update(cell_type):
    comm = MPI.COMM_WORLD
    N = 2
    mode = GhostMode.shared_facet
    num_cells = N * N * N * 6
    mesh = create_unit_cube(comm, N, N, N, ghost_mode=mode)
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)

    # If ghost_mode==shared_facet all interface facets are ghosts
    facets = compute_interface_facets(mesh.topology)
    facet_map = mesh.topology.index_map(tdim - 1)
    num_facets = facet_map.size_local
    interface_facets = numpy.where(facets)[0]
    assert numpy.all(interface_facets >= num_facets)

    # Remove all ghosts
    cell_map = mesh.topology.index_map(tdim)
    num_local_cells = cell_map.size_local
    dest = numpy.full([num_local_cells], fill_value=comm.rank, dtype=numpy.int32)
    mesh1 = update_ghosts(mesh, create_adjacencylist(dest))
    cell_map = mesh1.topology.index_map(tdim)
    assert cell_map.num_ghosts == 0
    assert cell_map.size_local == num_local_cells

    # Create mesh with maximal overlap (all cells are duplicated in all processes)
    dest = numpy.zeros([cell_map.size_local, comm.size], dtype=numpy.int32)
    ranks = numpy.arange(comm.size, dtype=numpy.int32)
    mask = numpy.ones(len(ranks), dtype=bool)
    mask[comm.rank] = False
    ranks = ranks[mask]
    ranks = numpy.insert(ranks, 0, comm.rank)
    dest[:, :] = ranks

    offset = numpy.arange(cell_map.size_local + 1, dtype=numpy.int32)
    offset = offset * comm.size

    cell_partitioning = create_adjacencylist(dest, offset)
    mesh2 = update_ghosts(mesh, cell_partitioning)
    cell_map2 = mesh2.topology.index_map(tdim)
    new_size = cell_map2.size_local + cell_map2.num_ghosts

    assert new_size == num_cells
    assert cell_map2.size_global == cell_map.size_global
    assert cell_map2.size_local == cell_map.size_local

    # No interface facets for maximal overlap
    mesh2.topology.create_connectivity(tdim - 1, tdim)
    facets = compute_interface_facets(mesh2.topology)
    assert not numpy.any(facets)
