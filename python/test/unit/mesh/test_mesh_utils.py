# Copyright (C) 2020 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import numpy as np
import pytest

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


def create_mesh_gmsh(shape, degree):
    import pygmsh
    geom = pygmsh.built_in.Geometry()

    if shape == cpp.mesh.CellType.triangle:
        # geom.add_rectangle(0.0, 2.0, 0.0, 1.0, 0.0, 2.1)
        geom = pygmsh.opencascade.Geometry()
        geom.add_disk([0.0, 0.0, 0.0], 1.0, char_length=0.2)
    elif shape == cpp.mesh.CellType.tetrahedron:
        print("1 ******")
        geom = pygmsh.opencascade.Geometry()
        geom.add_ball([0.0, 0.0, 0.0], 1.0, char_length=0.2)

    # rectangle = geom.add_rectangle(0.0, 1.0, 0.0, 1.0, 0.0, 20.1)
    # geom.add_raw_code("Recombine Surface {%s};" % rectangle.surface.id)

    if shape == cpp.mesh.CellType.triangle and degree == 1:
        mesh = pygmsh.generate_mesh(geom, dim=2, mesh_file_type="vtk")
        mesh.cells = [cells for cells in mesh.cells if cells.type == "triangle"]
    elif shape == cpp.mesh.CellType.triangle and degree == 2:
        mesh = pygmsh.generate_mesh(geom, dim=2, mesh_file_type="vtk", extra_gmsh_arguments=["-order", "2"])
        mesh.cells = [cells for cells in mesh.cells if cells.type == "triangle6"]
    elif shape == cpp.mesh.CellType.quadrilateral and degree == 1:
        mesh = pygmsh.generate_mesh(geom, dim=2, mesh_file_type="vtk")
        mesh.cells = [cells for cells in mesh.cells if cells.type == "quad"]
    elif shape == cpp.mesh.CellType.quadrilateral and degree == 2:
        mesh = pygmsh.generate_mesh(geom, dim=2, mesh_file_type="vtk", extra_gmsh_arguments=["-order", "2"])
        mesh.cells = [cells for cells in mesh.cells if cells.type == "quad9"]
    elif shape == cpp.mesh.CellType.tetrahedron and degree == 1:
        mesh = pygmsh.generate_mesh(geom, dim=3, mesh_file_type="vtk")
        mesh.cells = [cells for cells in mesh.cells if cells.type == "tetra"]
    elif shape == cpp.mesh.CellType.tetrahedron and degree == 2:
        mesh = pygmsh.generate_mesh(geom, dim=3, mesh_file_type="vtk", extra_gmsh_arguments=["-order", "2"])
        mesh.cells = [cells for cells in mesh.cells if cells.type == "tetra10"]

    # print("*3 *****", mesh.cells)
    # import meshio
    # meshio.write("test.vtu", mesh)

    points = mesh.points
    cells = mesh.cells[0].data
    return cells, points


# cells_in = [[6, 12, 2, 1, 11, 0], [12, 14, 7, 9, 10, 8], [7, 2, 12, 1, 10, 3], [6, 2, 13, 4, 5, 11]]

# Manual test
# cells_in = [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]]
# x = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
# x = np.array(x)

#     cells0 = [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]]

def get_layout(shape, degree):
    if shape == cpp.mesh.CellType.triangle and degree == 1:
        perms = np.zeros([5, 3], dtype=np.int8)
        perms[:] = [0, 1, 2]
        entity_dofs = [[set([0]), set([1]), set([2])], [set(), set(), set()], [set()]]
        return cpp.fem.ElementDofLayout(1, entity_dofs, [], [], shape, perms)
    elif shape == cpp.mesh.CellType.triangle and degree == 2:
        perms = np.zeros([5, 6], dtype=np.int8)
        perms[:] = [0, 1, 2, 3, 4, 5]
        entity_dofs = [[set([0]), set([1]), set([2])], [set([3]), set([4]), set([5])], [set()]]
        return cpp.fem.ElementDofLayout(1, entity_dofs, [], [], shape, perms)
    elif shape == cpp.mesh.CellType.quadrilateral and degree == 1:
        perms = np.zeros([6, 4], dtype=np.int8)
        perms[:] = [0, 1, 2, 3]
        entity_dofs = [[set([0]), set([1]), set([2]), set([3])], [set([]), set([]), set([]), set([])], [set()]]
        return cpp.fem.ElementDofLayout(1, entity_dofs, [], [], shape, perms)
    elif shape == cpp.mesh.CellType.quadrilateral and degree == 2:
        perms = np.zeros([6, 9], dtype=np.int8)
        perms[:] = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        entity_dofs = [[set([0]), set([1]), set([2]), set([3])], [set([4]), set([5]), set([6]), set([7])], [set([1])]]
        return cpp.fem.ElementDofLayout(1, entity_dofs, [], [], shape, perms)
    elif shape == cpp.mesh.CellType.tetrahedron and degree == 1:
        perms = np.zeros([18, 6], dtype=np.int8)
        perms[:] = [0, 1, 2, 3, 4, 5]
        entity_dofs = [[set([0]), set([1]), set([2]), set([3])], [set([]), set([]), set([]), set([]), set([]), set([])],
                       [set([]), set([]), set([]), set([])], [set()]]
        return cpp.fem.ElementDofLayout(1, entity_dofs, [], [], shape, perms)
    elif shape == cpp.mesh.CellType.tetrahedron and degree == 2:
        perms = np.zeros([18, 10], dtype=np.int8)
        perms[:] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        entity_dofs = [[set([0]), set([1]), set([2]), set([3])],
                       [set([4]), set([5]), set([6]), set([7]), set([8]), set([9])],
                       [set([]), set([]), set([]), set([])],
                       [set()]]
        return cpp.fem.ElementDofLayout(1, entity_dofs, [], [], shape, perms)
    else:
        raise RuntimeError("Unknown dof layout")


def test_topology_partition():
    """Test partitioning of cells"""
    # FIXME: make creating the ElementDofLayout simpler and clear

    pytest.importorskip("pygmsh")

    rank = cpp.MPI.rank(cpp.MPI.comm_world)
    size = cpp.MPI.size(cpp.MPI.comm_world)

    # Create mesh input data
    degree = 2
    # cell_type = cpp.mesh.CellType.triangle
    cell_type = cpp.mesh.CellType.tetrahedron
    # cell_type = cpp.mesh.CellType.quadrilateral
    layout = get_layout(cell_type, degree)
    dim = cpp.mesh.cell_dim(cell_type)

    if rank == 0:
        # Create mesh data
        cells, x = create_mesh_gmsh(cell_type, degree)
        print(cells)
        x = np.array(x[:, :dim])

        # Permute to DOLFIN ordering and create adjacency list
        cells = cpp.io.permute_cell_ordering(cells,
                                             cpp.io.permutation_vtk_to_dolfin(cell_type,
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
        x = np.zeros([0, dim])
        cells_v = cpp.graph.AdjacencyList64(0)

    # Compute the destination rank for cells on this process via graph
    # partitioning
    dest = cpp.mesh.partition_cells(cpp.MPI.comm_world, size, layout.cell_type, cells_v)
    assert len(dest) == cells_v.num_nodes

    # Distribute cells to destination rank
    cells, src, original_cell_index = cpp.mesh.distribute(cpp.MPI.comm_world, cells_v, dest)

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
    if cell_facet is not None:
        topology.set_connectivity(cell_facet, topology.dim, topology.dim - 1)
    if index_map is not None:
        topology.set_index_map(topology.dim - 1, index_map)
    if facet_vertex is not None:
        topology.set_connectivity(facet_vertex, topology.dim - 1, 0)
    facet_cell, _ = cpp.mesh.compute_connectivity(topology, topology.dim - 1, topology.dim)
    if facet_cell is not None:
        topology.set_connectivity(facet_cell, topology.dim - 1, topology.dim)

    cell_edge, edge_vertex, index_map = cpp.mesh.compute_entities(cpp.MPI.comm_world,
                                                                  topology, 1)
    if cell_edge is not None:
        topology.set_connectivity(cell_edge, topology.dim, 1)
    if index_map is not None:
        topology.set_index_map(1, index_map)
    if edge_vertex is not None:
        topology.set_connectivity(edge_vertex, 1, 0)

    # NOTE: Could be a local (MPI_COMM_SELF) dofmap?
    # Build 'geometry' dofmap on the topology
    dof_index_map, dofmap = cpp.fem.build_dofmap(cpp.MPI.comm_world,
                                                 topology, layout, 1)

    # Send/receive the 'cell nodes' (includes high-order geometry
    # nodes), and the global input cell index.
    #
    # NOTE: Maybe we can ensure that the 'global cells' are in the same
    # order as the owned cells (maybe they are already) to avoid the
    # need for global_index_nodes
    cell_nodes, global_index_cell = cpp.mesh.exchange(cpp.MPI.comm_world,
                                                      cells1, dest, set(src))
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
    x_g = coords[l2l]

    # Create Geometry
    geometry = cpp.mesh.Geometry(dof_index_map, dofmap, x_g, l2g, degree)

    print(coords.shape, geometry.num_points_global())

    # Create mesh
    mesh = cpp.mesh.Mesh(cpp.MPI.comm_world, topology, geometry)

    filename = os.path.join("mesh1.xdmf")
    encoding = XDMFFile.Encoding.HDF5
    with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(mesh)
    print("End")
