# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfinx
from dolfinx.cpp.mesh import GhostMode, CellType, partition_cells, partition_cells_kahip
from mpi4py import MPI
import pytest
import numpy as np


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


def test_custom_partitioner(): 
    pass


# if __name__ == "__main__":
#     Nx = 10
#     partitioner = partition_cells
#     tdim = mesh.topology.dim
#     mesh = dolfinx.BoxMesh(MPI.COMM_WORLD, [np.array([0, 0, 0]),
#                                             np.array([1, 1, 1])], [Nx, Nx, Nx], CellType.hexahedron,
#                            GhostMode.none, partitioner)
    
#     num_cells = mesh.topology.index_map(tdim).size_local
#     def custom_partitioner(mpi_comm, nparts, cell_type, cells, ghost_mode)
#         adj = numpy.full(num_cells, mpi_comm.rank, dtype=numpy.int32)
#         return adj
    
    
        
