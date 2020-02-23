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
    cell_type = cpp.mesh.CellType.triangle

    entity_dofs = [[set([0]), set([1]), set([2])], [set(), set(), set()], [set()]]
    layout = cpp.fem.ElementDofLayout(1, entity_dofs, [], [], cell_type, perms)
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
    layout = cpp.fem.ElementDofLayout(1, entity_dofs, [], [], cell_type, perms)
    rank = cpp.MPI.rank(cpp.MPI.comm_world)
    if rank == 0:
        cells1 = [[6, 12, 2, 1, 11, 0], [12, 14, 7, 9, 10, 8], [7, 2, 12, 1, 10, 3], [6, 2, 13, 4, 5, 11]]
        # cells1 = [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]]
        cells1 = cpp.graph.AdjacencyList64(cells1)
        cells_filtered1 = cpp.mesh.extract_topology(layout, cells1)
    else:
        cells_filtered1 = cpp.graph.AdjacencyList64(0)

    # Partition cells
    size = cpp.MPI.size(cpp.MPI.comm_world)
    dest = cpp.mesh.partition_cells(cpp.MPI.comm_world, size,
                                    layout.cell_type, cells_filtered1)
    assert len(dest) == cells_filtered1.num_nodes

    # Distribute cells
    cells, src = cpp.mesh.distribute(cpp.MPI.comm_world, cells_filtered1,
                                     dest)
    print("cells:", cells.num_nodes)

    # print(dest)
    assert cpp.MPI.sum(cpp.MPI.comm_world, cells.num_nodes) == 4

    # Build local cell connectivity
    cells_local, global_to_local_vertices, n = cpp.mesh.create_local_adjacency_list(cells)
    # if rank == 0:
    #     print(global_to_local_vertices)
    # print("Rank: ", rank, global_to_local_vertices)

    # Create topology and set cell-vertex topology
    topology = cpp.mesh.Topology(layout.cell_type)
    index_map = cpp.common.IndexMap(cpp.MPI.comm_self, cells_local.num_nodes, [], 1)
    topology.set_connectivity(cells_local, topology.dim, 0)
    topology.set_index_map(topology.dim, index_map)

    # Attach vertex IndexMap to local topology
    index_map = cpp.common.IndexMap(cpp.MPI.comm_self, n, [], 1)
    topology.set_index_map(0, index_map)

    # Create facets for local topology
    cell_facet, facet_vertex, index_map = cpp.mesh.compute_entities(cpp.MPI.comm_self,
                                                                    topology, topology.dim - 1)

    topology.set_connectivity(cell_facet, topology.dim, topology.dim - 1)
    if facet_vertex is not None:
        topology.set_connectivity(facet_vertex, topology.dim - 1, 0)
    topology.set_index_map(topology.dim - 1, index_map)

    facet_cell, _ = cpp.mesh.compute_connectivity(topology, topology.dim - 1, topology.dim)
    topology.set_connectivity(facet_cell, topology.dim - 1, topology.dim)

    # Get facets that are on the boundary
    boundary = cpp.mesh.compute_interior_facets(topology)
    topology.set_interior_facets(boundary)
    boundary = topology.on_boundary(topology.dim - 1)
    # print(boundary)

    # Build distributed cell-vertex AdjacencyList and IndexMap for
    # vertices

    print(topology.connectivity(2, 0).array())
    print(global_to_local_vertices)
    # return
    cells, vertex_map, l2g = cpp.mesh.create_distributed_adjacency_list(cpp.MPI.comm_world, topology,
                                                                        global_to_local_vertices)

    print("l2G", l2g)
    print("cells: ", cells.array())
    print("offset: ", cells.offsets())
    return
    print("Try", vertex_map.size_local)
    if rank == 1:
        print("test:", vertex_map.size_local, vertex_map.num_ghosts)

    # Build distributed topology
    num_cells = cpp.MPI.sum(cpp.MPI.comm_world, cells.num_nodes)
    print("num_cells", num_cells)
    topology = cpp.mesh.Topology(layout.cell_type)
    index_map = cpp.common.IndexMap(cpp.MPI.comm_self, num_cells, [], 1)
    topology.set_index_map(topology.dim, index_map)
    topology.set_connectivity(cells_local, topology.dim, 0)
    topology.set_index_map(0, vertex_map)

    c0 = cpp.graph.AdjacencyList(vertex_map.size_local + vertex_map.num_ghosts)
    topology.set_connectivity(c0, 0, 0)

    map0 = topology.index_map(0)
    if rank == 0:
        print("local", map0.size_local)
        print("global", map0.size_global)
        print("ghosts", map0.ghosts)
    print("ghosts owners", map0.ghost_owners())

    # test = topology.index_map(0)
    # print(test.size_local)
    v = topology.connectivity(0, 0)
    print(v.num_nodes)
    print(v.array())
    print(v.offsets())

    print(cells)
    c = topology.connectivity(2, 0)
    print(c.num_nodes)
    print(c.array())
    print(c.offsets())

    # Build dofmap
    dof_index_map, dofmap = cpp.fem.build_dofmap(cpp.MPI.comm_world,
                                                 topology, layout, 1)
