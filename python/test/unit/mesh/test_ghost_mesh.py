# Copyright (C) 2016 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest

from dolfin import MPI, UnitCubeMesh, UnitIntervalMesh, UnitSquareMesh, cpp

# See https://bitbucket.org/fenics-project/dolfin/issues/579


@pytest.mark.xfail(condition=MPI.size(MPI.comm_world) == 1,
                   reason="Shared ghost modes fail in serial")
def test_ghost_vertex_1d():
    mesh = UnitIntervalMesh(MPI.comm_world, 20,
                            ghost_mode=cpp.mesh.GhostMode.shared_vertex)
    assert mesh.num_entities_global(0) == 21
    assert mesh.num_entities_global(1) == 20


@pytest.mark.xfail(condition=MPI.size(MPI.comm_world) == 1,
                   reason="Shared ghost modes fail in serial")
def test_ghost_facet_1d():
    mesh = UnitIntervalMesh(MPI.comm_world, 20,
                            ghost_mode=cpp.mesh.GhostMode.shared_facet)
    assert mesh.num_entities_global(0) == 21
    assert mesh.num_entities_global(1) == 20


@pytest.mark.parametrize("mode", [pytest.param(cpp.mesh.GhostMode.shared_vertex,
                                               marks=pytest.mark.xfail(condition=MPI.size(MPI.comm_world) == 1,
                                                                       reason="Shared ghost modes fail in serial")),
                                  pytest.param(cpp.mesh.GhostMode.shared_facet,
                                               marks=pytest.mark.xfail(condition=MPI.size(MPI.comm_world) == 1,
                                                                       reason="Shared ghost modes fail in serial"))])
def test_ghost_2d(mode):
    N = 8
    num_cells = N * N * 2

    mesh = UnitSquareMesh(MPI.comm_world, N, N, ghost_mode=mode)
    if MPI.size(mesh.mpi_comm()) > 1:
        assert MPI.sum(mesh.mpi_comm(), mesh.num_cells()) > num_cells

    assert mesh.num_entities_global(0) == 81
    assert mesh.num_entities_global(2) == num_cells


@pytest.mark.parametrize("mode", [pytest.param(cpp.mesh.GhostMode.shared_vertex,
                                               marks=pytest.mark.xfail(condition=MPI.size(MPI.comm_world) == 1,
                                                                       reason="Shared ghost modes fail in serial")),
                                  pytest.param(cpp.mesh.GhostMode.shared_facet,
                                               marks=pytest.mark.xfail(condition=MPI.size(MPI.comm_world) == 1,
                                                                       reason="Shared ghost modes fail in serial"))])
def test_ghost_3d(mode):
    N = 2
    num_cells = N * N * N * 6

    mesh = UnitCubeMesh(MPI.comm_world, N, N, N, ghost_mode=mode)
    if MPI.size(mesh.mpi_comm()) > 1:
        assert MPI.sum(mesh.mpi_comm(), mesh.num_cells()) > num_cells

    assert mesh.num_entities_global(0) == 27
    assert mesh.num_entities_global(3) == num_cells


@pytest.mark.parametrize("mode", [cpp.mesh.GhostMode.none,
                                  pytest.param(cpp.mesh.GhostMode.shared_vertex,
                                               marks=pytest.mark.xfail(condition=MPI.size(MPI.comm_world) == 1,
                                                                       reason="Shared ghost modes fail in serial")),
                                  pytest.param(cpp.mesh.GhostMode.shared_facet,
                                               marks=pytest.mark.xfail(condition=MPI.size(MPI.comm_world) == 1,
                                                                       reason="Shared ghost modes fail in serial"))])
def test_ghost_connectivities(mode):
    # Ghosted mesh
    meshG = UnitSquareMesh(MPI.comm_world, 4, 4, ghost_mode=mode)
    meshG.create_connectivity(1, 2)

    # Reference mesh, not ghosted, not parallel
    meshR = UnitSquareMesh(MPI.comm_self, 4, 4, ghost_mode=cpp.mesh.GhostMode.none)
    meshR.create_connectivity(1, 2)
    tdim = meshR.topology.dim

    # Create reference mapping from facet midpoint to cell midpoint
    reference = {}
    facet_mp = cpp.mesh.midpoints(meshR, tdim - 1, range(meshR.num_entities(tdim - 1)))
    cell_mp = cpp.mesh.midpoints(meshR, tdim, range(meshR.num_entities(tdim)))
    reference = dict.fromkeys([tuple(row) for row in facet_mp], [])
    for i in range(meshR.num_entities(tdim - 1)):
        for cidx in meshR.topology.connectivity(1, 2).connections(i):
            reference[tuple(facet_mp[i])].append(cell_mp[cidx].tolist())

    # Loop through ghosted mesh and check connectivities
    tdim = meshG.topology.dim
    num_facets = meshG.num_entities(tdim - 1) - meshG.topology.ghost_offset(tdim - 1)
    allowable_cell_indices = range(meshG.num_cells())
    facet_mp = cpp.mesh.midpoints(meshG, tdim - 1, range(meshG.num_entities(tdim - 1)))
    cell_mp = cpp.mesh.midpoints(meshG, tdim, range(meshG.num_entities(tdim)))
    for i in range(num_facets):
        assert tuple(facet_mp[i]) in reference
        for cidx in meshG.topology.connectivity(1, 2).connections(i):
            assert cidx in allowable_cell_indices
            assert cell_mp[cidx].tolist() in reference[tuple(facet_mp[i])]
