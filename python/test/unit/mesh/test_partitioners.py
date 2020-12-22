# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI
import numpy as np
import pytest

import dolfinx
from dolfinx.cpp.mesh import GhostMode, CellType, partition_cells, partition_cells_kahip


@pytest.mark.parametrize("partitioner", [partition_cells,
                                         pytest.param(partition_cells_kahip,
                                                      marks=pytest.mark.skipif(not dolfinx.has_kahip,
                                                                               reason="Kahip is not available"))])
@pytest.mark.parametrize("Nx", [2, 5, 10])
@pytest.mark.parametrize("cell_type", [CellType.tetrahedron, CellType.hexahedron])
def test_partition_box_mesh(partitioner, Nx, cell_type):
    mesh = dolfinx.BoxMesh(MPI.COMM_WORLD, [np.array([0, 0, 0]),
                                            np.array([1, 1, 1])], [Nx, Nx, Nx], cell_type,
                           GhostMode.none, partitioner)
    tdim = mesh.topology.dim

    c = 6 if cell_type == CellType.tetrahedron else 1
    assert mesh.topology.index_map(tdim).size_global == Nx**3 * c
    assert mesh.topology.index_map(tdim).size_local != 0
    assert mesh.topology.index_map(0).size_global == (Nx + 1)**3


@pytest.mark.parametrize("Nx", [2, 5, 10])
@pytest.mark.parametrize("cell_type", [CellType.tetrahedron])
def xtest_custom_partitioner(Nx, cell_type):
    comm = MPI.COMM_WORLD
    points = [np.array([0, 0, 0]), np.array([1, 1, 1])]
    mesh = dolfinx.BoxMesh(MPI.COMM_WORLD, points, [Nx, Nx, Nx], cell_type, GhostMode.none)

    tdim = mesh.topology.dim
    num_local_cells = mesh.topology.index_map(tdim).size_local
    topo = mesh.geometry.dofmap.array.reshape(num_local_cells, mesh.geometry.dofmap.links(0).size)
    x = mesh.geometry.x
    domain = mesh.ufl_domain()

    # # Simple custom partitioner: keep cells in the current process
    # def custom_partitioner(mpi_comm, nparts, cell_type, cells, ghost_mode):
    #     dest = np.full(num_local_cells, mpi_comm.rank, np.int32)
    #     return dolfinx.cpp.graph.AdjacencyList_int32(dest)

    new_mesh = dolfinx.mesh.create_mesh(comm, topo, x, domain, GhostMode.none)

    # assert new_mesh.topology.index_map(tdim).size_local == mesh.topology.index_map(tdim).size_local
    assert new_mesh.topology.index_map(tdim).size_global == mesh.topology.index_map(tdim).size_global
