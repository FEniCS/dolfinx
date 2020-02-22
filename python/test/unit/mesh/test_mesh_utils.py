# Copyright (C) 2020 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np

from dolfinx import cpp
from dolfinx_utils.test.skips import skip_in_parallel


def test_extract_topology():
    """Test creation of topology adjacency lists, with extraction
    of cell vertices for 'higher-order' topologies"""

    # FIXME: make creating the ElementDofLayout simpler and clear
    perms = np.zeros([5, 3], dtype=np.int8)
    perms[:] = [0, 1, 2]
    entity_dofs = [[set([0]), set([1]), set([2])], [set(), set(),
                                                    set()], [set()]]
    layout = cpp.fem.ElementDofLayout(1, entity_dofs, [], [],
                                      cpp.mesh.CellType.triangle, perms)

    # Create cell 'topology' for 'P1' triangulation, i.e. no mid-side
    # nodes, and convert to an AdjacencyList
    cells0 = [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]]
    cells0 = cpp.graph.AdjacencyList64(cells0)

    # Extract a 'proper, vertex-only topology. Vertex indices are
    # unchanged. Should be same as input as input is vertex-only
    # triangulation.
    cells_filtered0 = cpp.mesh.extract_topology(layout, cells0)
    assert np.array_equal(cells0.array(), cells_filtered0.array())

    # Create element dof layout for P2 element
    entity_dofs = [[set([0]), set([1]), set([2])], [set([3]), set([4]), set([5])], [set()]]
    layout = cpp.fem.ElementDofLayout(1, entity_dofs, [], [],
                                      cpp.mesh.CellType.triangle, perms)

    # Create cell 'topology' for 'P2' triangulation, i.e. with mid-side
    # nodes, and convert to an AdjacencyList
    cells1 = [[0, 1, 4, 15, 14, 6], [0, 4, 3, 8, 9, 14], [1, 2, 5, 11, 12, 10], [1, 5, 4, 13, 15, 12]]
    cells1 = cpp.graph.AdjacencyList64(cells1)

    # Extract a 'proper, vertex-only topology. Vertex indices are
    # unchanged, and edges entries dropped
    cells_filtered1 = cpp.mesh.extract_topology(layout, cells1)
    assert np.array_equal(cells_filtered0.array(), cells_filtered1.array())


def test_topology_partition():
    """Test partitioning of cells"""
    # FIXME: make creating the ElementDofLayout simpler and clear

    rank = cpp.MPI.rank(cpp.MPI.comm_world)
    size = cpp.MPI.size(cpp.MPI.comm_world)

    cell_type = cpp.mesh.CellType.triangle

    # Create element dof layout for 'P1' simplex triangulation in 2D
    perms = np.zeros([5, 3], dtype=np.int8)
    perms[:] = [0, 1, 2]
    entity_dofs = [[set([0]), set([1]), set([2])], [set(), set(), set()], [set()]]
    layout = cpp.fem.ElementDofLayout(1, entity_dofs, [], [], cell_type, perms)

    # Create topology on rank 0, create empty AdjacencyList on other
    # ranks
    if rank == 0:
        cells0 = [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]]
        cells0 = cpp.graph.AdjacencyList64(cells0)
        cells_filtered0 = cpp.mesh.extract_topology(layout, cells0)
    else:
        cells_filtered0 = cpp.graph.AdjacencyList64(0)

    # Partition cells, compute the destination process for cells on this
    # process
    dest = cpp.mesh.partition_cells(cpp.MPI.comm_world, size,
                                    layout.cell_type, cells_filtered0)
    assert len(dest) == cells_filtered0.num_nodes

    # Create element dof layout for 'P2' simplex triangulation in 2D
    entity_dofs = [[set([0]), set([1]), set([2])], [set(), set(), set()], [set()]]
    layout = cpp.fem.ElementDofLayout(1, entity_dofs, [], [], cell_type, perms)
    if rank == 0:
        cells1 = [[6, 12, 2, 1, 11, 0], [12, 14, 7, 9, 10, 8], [7, 2, 12, 1, 10, 3], [6, 2, 13, 4, 5, 11]]
        # cells1 = [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]]
        cells1 = cpp.graph.AdjacencyList64(cells1)
        cells_filtered1 = cpp.mesh.extract_topology(layout, cells1)
    else:
        cells_filtered1 = cpp.graph.AdjacencyList64(0)

    # Partition cells, compute the destination process for cells on this
    # process
    dest = cpp.mesh.partition_cells(cpp.MPI.comm_world, size,
                                    layout.cell_type, cells_filtered1)
    assert len(dest) == cells_filtered1.num_nodes

    # Distribute cells to destination process
    cells, src = cpp.mesh.distribute(cpp.MPI.comm_world, cells_filtered1,
                                     dest)
    assert cpp.MPI.sum(cpp.MPI.comm_world, cells.num_nodes) == 4

    # print("cells:", cells.num_nodes)
    # print(dest)

    # Build local cell-vertex connectivity (with local vertex indices
    # [0, 1, 2, ..., n)), map from global indices in 'cells' to the
    # local vertex indices, and
    cells_local, global_to_local_vertices = cpp.mesh.create_local_adjacency_list(cells)
    assert len(global_to_local_vertices) == len(np.unique(cells.array()))
    assert len(global_to_local_vertices) == len(np.unique(cells_local.array()))
    assert np.unique(cells_local.array())[-1] == len(global_to_local_vertices) - 1

    # Create local topology, and set cell-vertex topology
    topology = cpp.mesh.Topology(layout.cell_type)
    index_map = cpp.common.IndexMap(cpp.MPI.comm_self, cells_local.num_nodes, [], 1)
    topology.set_connectivity(cells_local, topology.dim, 0)
    topology.set_index_map(topology.dim, index_map)

    # Attach vertex IndexMap to local topology
    n = len(global_to_local_vertices)
    index_map = cpp.common.IndexMap(cpp.MPI.comm_self, n, [], 1)
    topology.set_index_map(0, index_map)

    # Create facets for local topology, and attach to topology object
    cell_facet, facet_vertex, index_map = cpp.mesh.compute_entities(cpp.MPI.comm_self,
                                                                    topology, topology.dim - 1)
    topology.set_connectivity(cell_facet, topology.dim, topology.dim - 1)
    topology.set_index_map(topology.dim - 1, index_map)
    if facet_vertex is not None:
        topology.set_connectivity(facet_vertex, topology.dim - 1, 0)
    facet_cell, _ = cpp.mesh.compute_connectivity(topology, topology.dim - 1, topology.dim)
    topology.set_connectivity(facet_cell, topology.dim - 1, topology.dim)

    # Get facets that are on the boundary of the local topology, i.e are
    # connect to one cell only
    boundary = cpp.mesh.compute_interior_facets(topology)
    topology.set_interior_facets(boundary)
    boundary = topology.on_boundary(topology.dim - 1)

    # Build distributed cell-vertex AdjacencyList, IndexMap for
    # vertices, and map from local index to old global index
    cells, vertex_map, l2g = cpp.mesh.create_distributed_adjacency_list(cpp.MPI.comm_world, topology,
                                                                        global_to_local_vertices)

    # Create distributed topology
    topology = cpp.mesh.Topology(layout.cell_type)

    # Set vertex IndexMap and vertex-vertex connectivity
    topology.set_index_map(0, vertex_map)
    c0 = cpp.graph.AdjacencyList(vertex_map.size_local + vertex_map.num_ghosts)
    topology.set_connectivity(c0, 0, 0)

    # Set cell IndexMap and cell-vertex connectivity
    num_cells = cpp.MPI.sum(cpp.MPI.comm_world, cells.num_nodes)
    index_map = cpp.common.IndexMap(cpp.MPI.comm_self, num_cells, [], 1)
    topology.set_index_map(topology.dim, index_map)
    topology.set_connectivity(cells, topology.dim, 0)

    # Build 'geometry' dofmap on the topology
    dof_index_map, dofmap = cpp.fem.build_dofmap(cpp.MPI.comm_world,
                                                 topology, layout, 1)
