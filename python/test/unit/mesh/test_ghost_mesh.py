# Copyright (C) 2016 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
from dolfin import MPI, UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh, Cell, Cells, Facets, cpp

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
    meshG.init(1, 2)

    # Reference mesh, not ghosted, not parallel
    meshR = UnitSquareMesh(MPI.comm_self, 4, 4, ghost_mode=cpp.mesh.GhostMode.none)
    meshR.init(1, 2)

    # Create reference mapping from facet midpoint to cell midpoint
    reference = {}
    for facet in Facets(meshR):
        fidx = facet.index()
        facet_mp = tuple(facet.midpoint()[:])
        reference[facet_mp] = []
        for cidx in meshR.topology.connectivity(1, 2)(fidx):
            cell = Cell(meshR, cidx)
            cell_mp = tuple(cell.midpoint()[:])
            reference[facet_mp].append(cell_mp)

    # Loop through ghosted mesh and check connectivities
    allowable_cell_indices = [c.index() for c in Cells(meshG, cpp.mesh.MeshRangeType.ALL)]
    for facet in Facets(meshG, cpp.mesh.MeshRangeType.REGULAR):
        fidx = facet.index()
        facet_mp = tuple(facet.midpoint()[:])
        assert facet_mp in reference

        for cidx in meshG.topology.connectivity(1, 2)(fidx):
            assert cidx in allowable_cell_indices
            cell = Cell(meshG, cidx)
            cell_mp = tuple(cell.midpoint()[:])
            assert cell_mp in reference[facet_mp]
