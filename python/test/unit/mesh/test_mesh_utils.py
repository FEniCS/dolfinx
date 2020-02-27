# Copyright (C) 2020 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import numpy as np

from dolfinx import cpp
from dolfinx.io import XDMFFile


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
    cells = [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]]
    cells = cpp.graph.AdjacencyList64(cells)

    # Extract a 'proper, vertex-only topology. Vertex indices are
    # unchanged. Should be same as input as input is vertex-only
    # triangulation.
    cells_filtered = cpp.mesh.extract_topology(layout, cells)
    assert np.array_equal(cells.array(), cells_filtered.array())

    # Create element dof layout for P2 element
    perms = np.zeros([5, 6], dtype=np.int8)
    perms[:] = [0, 1, 2, 3, 4, 5]
    entity_dofs = [[set([0]), set([1]), set([2])], [set([3]), set([4]), set([5])], [set()]]
    layout = cpp.fem.ElementDofLayout(1, entity_dofs, [], [],
                                      cpp.mesh.CellType.triangle, perms)

    # Create cell 'topology' for 'P2' triangulation, i.e. with mid-side
    # nodes, and convert to an AdjacencyList
    cells = [[0, 1, 4, 15, 14, 6], [0, 4, 3, 8, 9, 14], [1, 2, 5, 11, 12, 10], [1, 5, 4, 13, 15, 12]]
    cells = cpp.graph.AdjacencyList64(cells)

    # Extract a 'proper, vertex-only topology. Vertex indices are
    # unchanged, and edges entries dropped
    cells_filtered1 = cpp.mesh.extract_topology(layout, cells)
    assert np.array_equal(cells_filtered.array(), cells_filtered1.array())


def create_mesh_gmsh(degree):
    import pygmsh
    geom = pygmsh.built_in.Geometry()
    geom.add_rectangle(0.0, 1.0, 0.0, 1.0, 0.0, 0.1)

    if degree == 1:
        mesh = pygmsh.generate_mesh(geom, mesh_file_type="vtk")
        mesh.cells = [cells for cells in mesh.cells if cells.type == "triangle"]
    else:
        mesh = pygmsh.generate_mesh(geom, mesh_file_type="vtk", extra_gmsh_arguments=["-order", "2"])
        mesh.cells = [cells for cells in mesh.cells if cells.type == "triangle6"]

    import meshio
    meshio.write("test.vtu", mesh)

    points = mesh.points
    cells = mesh.cells[0].data
    return cells, points


# cells_in = [[6, 12, 2, 1, 11, 0], [12, 14, 7, 9, 10, 8], [7, 2, 12, 1, 10, 3], [6, 2, 13, 4, 5, 11]]

# Manual test
# cells_in = [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]]
# x = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
# x = np.array(x)

#     cells0 = [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]]


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

    # Create element dof layout for 'P2' simplex triangulation in 2D
    perms = np.zeros([5, 6], dtype=np.int8)
    perms[:] = [0, 1, 2, 3, 4, 5]
    entity_dofs = [[set([0]), set([1]), set([2])], [set([3]), set([4]), set([5])], [set()]]
    layout = cpp.fem.ElementDofLayout(1, entity_dofs, [], [], cell_type, perms)

    # Create mesh input data
    degree = 2
    if rank == 0:
        # Create mesh data
        cells, x = create_mesh_gmsh(degree)
        x = np.array(x[:, :2])

        # Permute to DOLFIN ordering and create adjacency list
        cells = cpp.io.permute_cell_ordering(cells,
                                             cpp.io.permutation_vtk_to_dolfin(cpp.mesh.CellType.triangle,
                                                                              cells.shape[1]))
        cells1 = cpp.graph.AdjacencyList64(cells)

        # Extract topology data, e.g. just the vertices. For P1 geometry
        # this should just be the identity operator. For other elements
        # the filtered lists may have 'gaps', i.e. the indices might not
        # be contiguous.
        cells_v = cpp.mesh.extract_topology(layout, cells1)
    else:
        # Empty data on ranks other than 0
        cells1 = cpp.graph.AdjacencyList64(0)
        x = np.zeros([0, 2])
        cells_v = cpp.graph.AdjacencyList64(0)

    # Compute the destination rank for cells on this process via graph
    # partitioning
    dest = cpp.mesh.partition_cells(cpp.MPI.comm_world, size,
                                    layout.cell_type, cells_v)
    assert len(dest) == cells_v.num_nodes

    # Distribute cells to destination rank
    cells, src, original_cell_index = cpp.mesh.distribute(cpp.MPI.comm_world, cells_v,
                                                          dest)

    # Build local cell-vertex connectivity, with local vertex indices
    # [0, 1, 2, ..., n), and get map from global vertex indices in
    # 'cells' to the local vertex indices
    cells_local, local_to_global_vertices = cpp.mesh.create_local_adjacency_list(cells)
    assert len(local_to_global_vertices) == len(np.unique(cells.array()))
    assert len(local_to_global_vertices) == len(np.unique(cells_local.array()))
    assert np.unique(cells_local.array())[-1] == len(local_to_global_vertices) - 1

    # Create local topology, create IndexMap for cells, and set cell-vertex topology
    topology_local = cpp.mesh.Topology(layout.cell_type)
    index_map = cpp.common.IndexMap(cpp.MPI.comm_self, cells_local.num_nodes, [], 1)
    topology_local.set_index_map(topology_local.dim, index_map)
    topology_local.set_connectivity(cells_local, topology_local.dim, 0)

    # Attach vertex IndexMap to local topology
    n = len(local_to_global_vertices)
    index_map = cpp.common.IndexMap(cpp.MPI.comm_self, n, [], 1)
    topology_local.set_index_map(0, index_map)

    # Create facets for local topology, and attach to topology object
    cell_facet, facet_vertex, index_map = cpp.mesh.compute_entities(cpp.MPI.comm_self,
                                                                    topology_local, topology_local.dim - 1)
    topology_local.set_connectivity(cell_facet, topology_local.dim, topology_local.dim - 1)
    topology_local.set_index_map(topology_local.dim - 1, index_map)
    if facet_vertex is not None:
        topology_local.set_connectivity(facet_vertex, topology_local.dim - 1, 0)
    facet_cell, _ = cpp.mesh.compute_connectivity(topology_local, topology_local.dim - 1, topology_local.dim)
    topology_local.set_connectivity(facet_cell, topology_local.dim - 1, topology_local.dim)

    # Get facets that are on the boundary of the local topology, i.e are
    # connect to one cell only
    boundary = cpp.mesh.compute_interior_facets(topology_local)
    topology_local.set_interior_facets(boundary)
    boundary = topology_local.on_boundary(topology_local.dim - 1)

    # Build distributed cell-vertex AdjacencyList, IndexMap for
    # vertices, and map from local index to old global index
    cells, vertex_map = cpp.mesh.create_distributed_adjacency_list(cpp.MPI.comm_world, topology_local,
                                                                   local_to_global_vertices)

    # --- Create distributed topology
    topology = cpp.mesh.Topology(layout.cell_type)

    # Set vertex IndexMap, and vertex-vertex connectivity
    topology.set_index_map(0, vertex_map)
    c0 = cpp.graph.AdjacencyList(vertex_map.size_local + vertex_map.num_ghosts)
    topology.set_connectivity(c0, 0, 0)

    # Set cell IndexMap and cell-vertex connectivity
    index_map = cpp.common.IndexMap(cpp.MPI.comm_world, cells.num_nodes, [], 1)
    topology.set_index_map(topology.dim, index_map)
    topology.set_connectivity(cells, topology.dim, 0)

    # Create facets for topology, and attach to topology object
    cell_facet, facet_vertex, index_map = cpp.mesh.compute_entities(cpp.MPI.comm_world,
                                                                    topology, topology.dim - 1)
    topology.set_connectivity(cell_facet, topology.dim, topology.dim - 1)
    topology.set_index_map(topology.dim - 1, index_map)
    if facet_vertex is not None:
        topology.set_connectivity(facet_vertex, topology.dim - 1, 0)
    facet_cell, _ = cpp.mesh.compute_connectivity(topology, topology.dim - 1, topology.dim)
    topology.set_connectivity(facet_cell, topology.dim - 1, topology.dim)

    # NOTE: Could be a local (MPI_COMM_SELF) dofmap?
    # Build 'geometry' dofmap on the topology
    dof_index_map, dofmap = cpp.fem.build_dofmap(cpp.MPI.comm_world,
                                                 topology, layout, 1)

    print("Dofmap")
    print(dofmap)

    # Send/receive the 'cell nodes' (includes high-order geometry
    # nodes), and the global input cell index.
    #
    # NOTE: Maybe we can ensure that the 'global cells' are in the same
    # order as the owned cells (maybe they are already) to avoid the
    # need for global_index_nodes
    cell_nodes, global_index_cell = cpp.mesh.exchange(cpp.MPI.comm_world,
                                                      cells1, dest, set(src))

    # print("cell_nodes")
    # print(cell_nodes)
    # for n in range(cell_nodes.num_nodes):
    #     print("  ", cell_nodes.links(n))
    # print("End cell_nodes")

    assert cell_nodes.num_nodes == cells.num_nodes
    assert global_index_cell == original_cell_index

    # Check that number of dofs is equal to number of 'nodes' in the input
    assert dofmap.array().shape == cell_nodes.array().shape

    # Build list of unique node indices for adjacency list
    indices = np.unique(cell_nodes.array())

    l2g = cpp.mesh.compute_local_to_global_links(cell_nodes, dofmap)
    l2l = cpp.mesh.compute_local_to_local(l2g, indices)

    # Fetch node coordinates
    coords = cpp.mesh.fetch_data(cpp.MPI.comm_world, indices, x)

    # Build dof array
    x_g = np.zeros([len(l2l), 2])
    for i, d in enumerate(l2l):
        x_g[i] = coords[d]
    print("-------")

    # Create Geometry
    geometry = cpp.mesh.Geometry(dof_index_map, dofmap, x_g, l2g, degree)

    # Create mesh
    mesh = cpp.mesh.Mesh(cpp.MPI.comm_world, topology, geometry)

    filename = os.path.join("mesh1.xdmf")
    encoding = XDMFFile.Encoding.HDF5
    with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(mesh)
