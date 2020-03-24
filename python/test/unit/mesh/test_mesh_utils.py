# Copyright (C) 2020 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import numpy as np
import pytest
from dolfinx_utils.test.fixtures import tempdir

from dolfinx import cpp
from dolfinx.io import XDMFFile

assert (tempdir)


def xtest_extract_topology():
    """Test creation of topology adjacency lists, with extraction
    of cell vertices for 'higher-order' topologies"""

    # FIXME: make creating the ElementDofLayout simpler and clear
    perms = np.zeros([3, 3], dtype=np.int8)
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
    perms = np.zeros([3, 6], dtype=np.int8)
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


def create_mesh_gmsh(shape, order):
    """Compute cell topology and geometric points for a range of cells types
    and geometric orders

    """

    import pygmsh
    geom = pygmsh.built_in.Geometry()
    if shape == cpp.mesh.CellType.triangle:
        geom = pygmsh.opencascade.Geometry()
        geom.add_disk([0.0, 0.0, 0.0], 1.0, char_length=1.2)
    elif shape == cpp.mesh.CellType.tetrahedron:
        geom = pygmsh.opencascade.Geometry()
        geom.add_ball([0.0, 0.0, 0.0], 1.0, char_length=0.2)
    elif shape == cpp.mesh.CellType.quadrilateral:
        rect = geom.add_rectangle(0.0, 2.0, 0.0, 1.0, 0.0, 0.1)
        geom.set_recombined_surfaces([rect.surface])
    elif shape == cpp.mesh.CellType.hexahedron:
        lbw = [2, 3, 5]
        points = [geom.add_point([x, 0.0, 0.0], 1.0) for x in [0.0, lbw[0]]]
        line = geom.add_line(*points)
        _, rectangle, _ = geom.extrude(line, translation_axis=[0.0, lbw[1], 0.0], num_layers=lbw[1], recombine=True)
        geom.extrude(rectangle, translation_axis=[0.0, 0.0, lbw[2]], num_layers=lbw[2], recombine=True)

    names = {
        (cpp.mesh.CellType.triangle, 1): "triangle",
        (cpp.mesh.CellType.triangle, 2): "triangle6",
        (cpp.mesh.CellType.quadrilateral, 1): "quad",
        (cpp.mesh.CellType.quadrilateral, 2): "quad9",
        (cpp.mesh.CellType.tetrahedron, 1): "tetra",
        (cpp.mesh.CellType.tetrahedron, 2): "tetra10",
        (cpp.mesh.CellType.hexahedron, 1): "hexahedron",
        (cpp.mesh.CellType.hexahedron, 2): "hexahedron27"
    }

    # Generate mesh
    dim = cpp.mesh.cell_dim(shape)
    mesh = pygmsh.generate_mesh(geom, dim=dim, mesh_file_type="vtk",
                                extra_gmsh_arguments=["-order", "{}".format(order)])
    name = names[(shape, order)]
    cells = np.array([cells for cells in mesh.cells if cells.type == name])

    return cells[0][1], mesh.points


def get_dof_layout(shape, order):
    """Create ElementDofLayouts for a range of Lagrange element types"""
    if shape == cpp.mesh.CellType.triangle and order == 1:
        perms = np.zeros([3, 3], dtype=np.int8)
        perms[:] = range(3)
        entity_dofs = [[set([0]), set([1]), set([2])], 3 * [set()], [set()]]
        return cpp.fem.ElementDofLayout(1, entity_dofs, [], [], shape, perms)
    elif shape == cpp.mesh.CellType.triangle and order == 2:
        perms = np.zeros([3, 6], dtype=np.int8)
        perms[:] = range(6)
        entity_dofs = [[set([0]), set([1]), set([2])], [set([3]), set([4]), set([5])], [set()]]
        return cpp.fem.ElementDofLayout(1, entity_dofs, [], [], shape, perms)
    elif shape == cpp.mesh.CellType.quadrilateral and order == 1:
        perms = np.zeros([4, 4], dtype=np.int8)
        perms[:] = range(4)
        entity_dofs = [[set([0]), set([1]), set([2]), set([3])],
                       4 * [set()], [set()]]
        return cpp.fem.ElementDofLayout(1, entity_dofs, [], [], shape, perms)
    elif shape == cpp.mesh.CellType.quadrilateral and order == 2:
        perms = np.zeros([4, 9], dtype=np.int8)
        perms[:] = range(9)
        entity_dofs = [[set([0]), set([1]), set([3]), set([4])],
                       [set([2]), set([5]), set([6]), set([7])],
                       [set([8])]]
        return cpp.fem.ElementDofLayout(1, entity_dofs, [], [], shape, perms)
    elif shape == cpp.mesh.CellType.tetrahedron and order == 1:
        perms = np.zeros([14, 4], dtype=np.int8)
        perms[:] = range(4)
        entity_dofs = [[set([0]), set([1]), set([2]), set([3])],
                       6 * [set()], 4 * [set()], [set()]]
        return cpp.fem.ElementDofLayout(1, entity_dofs, [], [], shape, perms)
    elif shape == cpp.mesh.CellType.tetrahedron and order == 2:
        perms = np.zeros([14, 10], dtype=np.int8)
        perms[:] = range(10)
        entity_dofs = [[set([0]), set([1]), set([2]), set([3])],
                       [set([4]), set([5]), set([6]), set([7]), set([8]), set([9])],
                       4 * [set()], [set()]]
        return cpp.fem.ElementDofLayout(1, entity_dofs, [], [], shape, perms)
    elif shape == cpp.mesh.CellType.hexahedron and order == 1:
        perms = np.zeros([24, 8], dtype=np.int8)
        perms[:] = range(8)
        entity_dofs = [
            [set([0]), set([1]), set([2]), set([3]), set([4]), set([5]), set([6]), set([7])],
            12 * [set()], 6 * [set()], [set([])]
        ]
        return cpp.fem.ElementDofLayout(1, entity_dofs, [], [], shape, perms)
    elif shape == cpp.mesh.CellType.hexahedron and order == 2:
        perms = np.zeros([24, 27], dtype=np.int8)
        perms[:] = range(27)
        entity_dofs = [
            [set([0]), set([1]), set([3]), set([4]), set([9]), set([10]), set([12]), set([13])],
            [set([2]), set([5]), set([11]), set([14]), set([6]), set([7]), set([15]), set([16]),
                set([18]), set([19]), set([21]), set([22])],
            [set([8]), set([17]), set([20]), set([23]), set([24]), set([25])],
            [set([26])]
        ]
        return cpp.fem.ElementDofLayout(1, entity_dofs, [], [], shape, perms)
    else:
        raise RuntimeError("Unknown dof layout")


@pytest.mark.parametrize("order", [
    1,
    2
])
@pytest.mark.parametrize("shape", [
    cpp.mesh.CellType.triangle,
    cpp.mesh.CellType.quadrilateral,
    cpp.mesh.CellType.tetrahedron,
    cpp.mesh.CellType.hexahedron
])
def test_topology_partition(tempdir, shape, order):
    """Test partitioning and creation of meshes"""

    pytest.importorskip("pygmsh")

    size = cpp.MPI.size(cpp.MPI.comm_world)
    layout = get_dof_layout(shape, order)
    dim = cpp.mesh.cell_dim(shape)

    # rank = cpp.MPI.rank(cpp.MPI.comm_world)
    # if rank == 0:
    #     # Create mesh data
    #     cells, x = create_mesh_gmsh(shape, order)
    #     x = np.array(x[:, : dim])

    #     # Permute to DOLFIN ordering and create adjacency list
    #     cells = cpp.io.permute_cell_ordering(cells,
    #                                           cpp.io.permutation_vtk_to_dolfin(shape,
    #                                                                            cells.shape[1]))
    #     cells_global = cpp.graph.AdjacencyList64(cells)

    #     # Extract topology data, e.g. just the vertices. For P1 geometry
    #     # this should just be the identity operator. For other elements
    #     # the filtered lists may have 'gaps', i.e. the indices might not
    #     # be contiguous.
    #     cells_global_v = cpp.mesh.extract_topology(layout, cells_global)
    # else:
    #     # Empty data on ranks other than 0
    #     cells_global = cpp.graph.AdjacencyList64(0)
    #     x = np.zeros([0, dim])
    #     cells_global_v = cpp.graph.AdjacencyList64(0)

    # Create mesh data
    cells, x = create_mesh_gmsh(shape, order)

    # Divide data amongst ranks (for testing). Possible to start will
    # all data on a single rank.
    range_c = cpp.MPI.local_range(cpp.MPI.comm_world, len(cells))
    range_v = cpp.MPI.local_range(cpp.MPI.comm_world, len(x))
    cells = cells[range_c[0]:range_c[1]]
    x = np.array(x[range_v[0]:range_v[1], : dim])

    # Permute to DOLFIN ordering and create adjacency list
    cells = cpp.io.permute_cell_ordering(cells,
                                         cpp.io.permutation_vtk_to_dolfin(shape,
                                                                          cells.shape[1]))
    cells_global = cpp.graph.AdjacencyList64(cells)

    # Extract topology data, e.g. just the vertices. For P1 geometry
    # this should just be the identity operator. For other elements
    # the filtered lists may have 'gaps', i.e. the indices might not
    # be contiguous.
    cells_global_v = cpp.mesh.extract_topology(layout, cells_global)

    # Compute the destination rank for cells on this process via graph
    # partitioning
    dest = cpp.mesh.partition_cells(cpp.MPI.comm_world, size, layout.cell_type,
                                    cells_global_v, cpp.mesh.GhostMode.none)
    assert len(dest) == cells_global_v.num_nodes

    # Distribute cells to destination rank
    cells, src, original_cell_index, ghost_index = cpp.graph.distribute(cpp.MPI.comm_world,
                                                                        cells_global_v, dest)

    # Build local cell-vertex connectivity, with local vertex indices
    # [0, 1, 2, ..., n), from cell-vertex connectivity using global
    # indices and get map from global vertex indices in 'cells' to the
    # local vertex indices
    cells_local, local_to_global_vertices = cpp.graph.create_local_adjacency_list(cells)
    assert len(local_to_global_vertices) == len(np.unique(cells.array()))
    assert len(local_to_global_vertices) == len(np.unique(cells_local.array()))
    assert np.unique(cells_local.array())[-1] == len(local_to_global_vertices) - 1

    # Create (i) local topology object and (ii) IndexMap for cells, and
    # set cell-vertex topology
    topology_local = cpp.mesh.Topology(layout.cell_type)
    tdim = topology_local.dim
    map = cpp.common.IndexMap(cpp.MPI.comm_self, cells_local.num_nodes, [], 1)
    topology_local.set_index_map(tdim, map)
    topology_local.set_connectivity(cells_local, tdim, 0)

    # Attach an IndexMap for vertices to local topology
    n = len(local_to_global_vertices)
    index_map = cpp.common.IndexMap(cpp.MPI.comm_self, n, [], 1)
    topology_local.set_index_map(0, index_map)

    # Create facets for local topology, and attach to the topology object
    cf, fv, map = cpp.mesh.compute_entities(cpp.MPI.comm_self,
                                            topology_local, tdim - 1)
    topology_local.set_connectivity(cf, tdim, tdim - 1)
    topology_local.set_index_map(tdim - 1, index_map)
    if fv is not None:
        topology_local.set_connectivity(fv, tdim - 1, 0)
    fc, _ = cpp.mesh.compute_connectivity(topology_local, tdim - 1, tdim)
    topology_local.set_connectivity(fc, tdim - 1, tdim)

    # Get facets that are on the boundary of the local topology, i.e are
    # connect to one cell only
    boundary = cpp.mesh.compute_interior_facets(topology_local)
    topology_local.set_interior_facets(boundary)
    boundary = topology_local.on_boundary(tdim - 1)

    # Build distributed cell-vertex AdjacencyList, IndexMap for
    # vertices, and map from local index to old global index
    exterior_vertices = cpp.mesh.compute_vertex_exterior_markers(topology_local)
    cells, vertex_map = cpp.graph.create_distributed_adjacency_list(cpp.MPI.comm_world, cells_local,
                                                                    local_to_global_vertices, exterior_vertices)

    # --- Create distributed topology
    topology = cpp.mesh.Topology(layout.cell_type)

    # Set vertex IndexMap, and vertex-vertex connectivity
    topology.set_index_map(0, vertex_map)
    c0 = cpp.graph.AdjacencyList(vertex_map.size_local + vertex_map.num_ghosts)
    topology.set_connectivity(c0, 0, 0)

    # Set cell IndexMap and cell-vertex connectivity
    index_map = cpp.common.IndexMap(cpp.MPI.comm_world, cells.num_nodes, [], 1)
    topology.set_index_map(tdim, index_map)
    topology.set_connectivity(cells, tdim, 0)

    # Create facets for topology, and attach to topology object
    cf, fv, index_map = cpp.mesh.compute_entities(cpp.MPI.comm_world,
                                                  topology, tdim - 1)
    if cf is not None:
        topology.set_connectivity(cf, tdim, tdim - 1)
    if index_map is not None:
        topology.set_index_map(tdim - 1, index_map)
    if fv is not None:
        topology.set_connectivity(fv, tdim - 1, 0)
    fc, _ = cpp.mesh.compute_connectivity(topology, tdim - 1, tdim)
    if fc is not None:
        topology.set_connectivity(fc, tdim - 1, tdim)

    ce, ev, index_map = cpp.mesh.compute_entities(cpp.MPI.comm_world,
                                                  topology, 1)
    if ce is not None:
        topology.set_connectivity(ce, tdim, 1)
    if index_map is not None:
        topology.set_index_map(1, index_map)
    if ev is not None:
        topology.set_connectivity(ev, 1, 0)

    # --- Geometry

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
    #
    # NOTE: This could be optimised as we have earlier computed which
    # processes own the cells this process needs.
    cell_nodes, src, global_index_cell, ghost_owners = \
        cpp.graph.distribute(cpp.MPI.comm_world,
                             cells_global, dest)
    assert cell_nodes.num_nodes == cells.num_nodes
    assert global_index_cell == original_cell_index

    # Check that number of dofs is equal to number of geometry 'nodes'
    # from the mesh data input
    assert dofmap.array().shape == cell_nodes.array().shape

    # Build list of unique (global) node indices from adjacency list
    # (geometry nodes)
    indices = np.unique(cell_nodes.array())

    # Fetch node coordinates by global index from other ranks. Order of
    # coords matches order of the indices in 'indices'
    coords = cpp.graph.distribute_data(cpp.MPI.comm_world, indices, x)

    # Compute local-to-global map from local indices in dofmap to the
    # corresponding global indices in cell_nodes
    l2g = cpp.graph.compute_local_to_global_links(cell_nodes, dofmap)

    # Compute local (dof) to local (position in coords) map from (i)
    # local-to-global for dofs and (ii) local-to-global for entries in
    # coords
    l2l = cpp.graph.compute_local_to_local(l2g, indices)

    # Build coordinate dof array
    x_g = coords[l2l]

    # Create Geometry
    geometry = cpp.mesh.Geometry(dof_index_map, dofmap, layout, x_g, l2g)

    # Create mesh
    mesh = cpp.mesh.Mesh(cpp.MPI.comm_world, topology, geometry)

    # Write mesh to file
    filename = os.path.join(tempdir, "mesh_{}_{}.xdmf".format(cpp.mesh.to_string(shape), order))
    # print(filename)
    encoding = XDMFFile.Encoding.HDF5
    with XDMFFile(mesh.mpi_comm(), filename, encoding=encoding) as file:
        file.write(mesh)
