# Copyright (C) 2020 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np

from dolfinx import cpp
from dolfinx_utils.test.skips import skip_in_parallel


@skip_in_parallel
def test_extract_topology():
    """Test extract of cell vertices"""
    # FIXME: make creating the ElementDofLayout simpler and clear
    perms = np.zeros([5, 3], dtype=np.int8)
    perms[:] = [0, 1, 2]

    entity_dofs = [[set([0]), set([1]), set([2])], [set(), set(),
                                                    set()], [set()]]
    cells0 = [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]]
    cells0 = cpp.graph.AdjacencyList64(cells0)
    layout = cpp.fem.ElementDofLayout(1, entity_dofs, [], [], cpp.mesh.CellType.triangle, perms)
    cells_filtered0 = cpp.mesh.extract_topology(layout, cells0)
    assert np.array_equal(cells0.array(), cells_filtered0.array())

    entity_dofs = [[set([0]), set([1]), set([2])], [set([3]), set([4]), set([5])], [set()]]
    cells1 = [[0, 1, 4, 15, 14, 6], [0, 4, 3, 8, 9, 14], [1, 2, 5, 11, 12, 10], [1, 5, 4, 13, 15, 12]]
    cells1 = cpp.graph.AdjacencyList64(cells1)
    layout = cpp.fem.ElementDofLayout(1, entity_dofs, [], [], cpp.mesh.CellType.triangle, perms)
    cells_filtered1 = cpp.mesh.extract_topology(layout, cells1)
    assert np.array_equal(cells_filtered0.array(), cells_filtered1.array())


def test_partition():
    """Test partitioning of cells"""
    # FIXME: make creating the ElementDofLayout simpler and clear

    perms = np.zeros([5, 3], dtype=np.int8)
    perms[:] = [0, 1, 2]

    entity_dofs = [[set([0]), set([1]), set([2])], [set(), set(), set()], [set()]]
    layout = cpp.fem.ElementDofLayout(1, entity_dofs, [], [], cpp.mesh.CellType.triangle, perms)
    rank = cpp.MPI.rank(cpp.MPI.comm_world)
    if rank == 0:
        cells0 = [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]]
        cells0 = cpp.graph.AdjacencyList64(cells0)
        cells_filtered0 = cpp.mesh.extract_topology(layout, cells0)
    else:
        cells_filtered0 = cpp.graph.AdjacencyList64(0)
    size = cpp.MPI.size(cpp.MPI.comm_world)
    dest = cpp.mesh.partition_cells(cpp.MPI.comm_world, size,
                                    layout.cell_type, cells_filtered0)

    entity_dofs = [[set([0]), set([1]), set([2])], [set(), set(), set()], [set()]]
    layout = cpp.fem.ElementDofLayout(1, entity_dofs, [], [], cpp.mesh.CellType.triangle, perms)
    rank = cpp.MPI.rank(cpp.MPI.comm_world)
    if rank == 0:
        cells1 = [[6, 12, 2, 1, 11, 0], [12, 14, 7, 9, 10, 8], [7, 2, 12, 1, 10, 3], [6, 2, 13, 4, 5, 11]]
        cells1 = cpp.graph.AdjacencyList64(cells1)
        cells_filtered1 = cpp.mesh.extract_topology(layout, cells1)
    else:
        cells_filtered1 = cpp.graph.AdjacencyList64(0)

    size = cpp.MPI.size(cpp.MPI.comm_world)
    dest = cpp.mesh.partition_cells(cpp.MPI.comm_world, size,
                                    layout.cell_type, cells_filtered1)
    assert len(dest) == cells_filtered1.num_nodes

    cells, src = cpp.mesh.distribute(cpp.MPI.comm_world, cells_filtered1,
                                     dest)
    assert cpp.MPI.sum(cpp.MPI.comm_world, cells.num_nodes) == 4
