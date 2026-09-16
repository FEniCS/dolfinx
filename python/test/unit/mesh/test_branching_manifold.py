# Copyright (C) 2025 Paul T. Kühner and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest

import basix
import ufl
from dolfinx.graph import adjacencylist, partitioner
from dolfinx.mesh import (
    CellType,
    GhostMode,
    cell_num_vertices,
    compute_midpoints,
    create_mesh,
    create_unit_cube,
    create_unit_square,
    entities_to_geometry,
    exterior_facet_indices,
)

_graph_partitioners: list = []
try:
    from dolfinx.graph import partitioner_scotch

    _graph_partitioners.append(partitioner_scotch())
except ImportError:
    _graph_partitioners.append(
        pytest.param(None, marks=pytest.mark.skip(reason="DOLFINx build without SCOTCH"))
    )
try:
    from dolfinx.graph import partitioner_parmetis

    _graph_partitioners.append(partitioner_parmetis())
except ImportError:
    _graph_partitioners.append(
        pytest.param(None, marks=pytest.mark.skip(reason="DOLFINx built without Parmetis"))
    )
try:
    from dolfinx.graph import partitioner_kahip

    _graph_partitioners.append(partitioner_kahip())
except ImportError:
    _graph_partitioners.append(
        pytest.param(None, marks=pytest.mark.skip(reason="DOLFINx built without KaHiP"))
    )


@pytest.mark.parametrize(
    "dim,cell_type",
    [
        (2, CellType.triangle),
        (2, CellType.quadrilateral),
        (3, CellType.hexahedron),
        (3, CellType.tetrahedron),
    ],
)
def test_edge_skeleton_mesh(dim, cell_type):
    """Creates the edge skeleton mesh of a regular unit square/cube and checks for correct
    connectivity information.

    The edge skeleton mesh is the mesh formed by the edges of another mesh (edges -> cell). In
    particular this is a branching mesh.
    """
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        if dim == 2:
            mesh = create_unit_square(MPI.COMM_SELF, 4, 4, cell_type=cell_type)
        else:
            mesh = create_unit_cube(MPI.COMM_SELF, 2, 2, 2, cell_type=cell_type)

        top = mesh.topology
        top.create_connectivity(1, 0)
        e_to_v = top.connectivity(1, 0)
        new_x = mesh.geometry.x[:, :-1] if dim == 2 else mesh.geometry.x
        cells = e_to_v.array.reshape(-1, 2)
    else:
        new_x = np.empty((0, dim), dtype=np.float64)
        cells = np.empty((0, dim), dtype=np.int64)

    element = ufl.Mesh(basix.ufl.element("Lagrange", "interval", 1, shape=(dim,)))

    if cell_type == CellType.quadrilateral:
        max_facet_to_cell_links = 4
    elif cell_type == CellType.triangle:
        max_facet_to_cell_links = 6
    elif cell_type == CellType.hexahedron:
        max_facet_to_cell_links = 6
    elif cell_type == CellType.tetrahedron:
        max_facet_to_cell_links = 14

    skeleton_mesh = create_mesh(
        comm,
        cells,
        element,
        new_x,
        partitioner(),
        ghost_mode=GhostMode.shared_facet,
        max_facet_to_cell_links=max_facet_to_cell_links,
    )

    skeleton_top = skeleton_mesh.topology
    skeleton_top.create_connectivity(0, 1)
    skeleton_f_to_c = skeleton_top.connectivity(0, 1)

    skeleton_im_f = skeleton_mesh.topology.index_map(0)

    def on_boundary(x):
        return np.any(np.isclose(x[:dim], 0)) or np.any(np.isclose(x[:dim], 1))

    for facet in range(skeleton_im_f.size_local):
        matched = len(skeleton_f_to_c.links(facet)) == max_facet_to_cell_links
        assert matched or on_boundary(skeleton_mesh.geometry.x[facet])


@pytest.mark.parametrize("cell_type", [CellType.hexahedron, CellType.tetrahedron])
def test_facet_skeleton_mesh(cell_type):
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        mesh = create_unit_cube(MPI.COMM_SELF, 4, 4, 4, cell_type=cell_type)

        top = mesh.topology
        top.create_connectivity(2, 0)
        tdim = top.dim
        facet_map = mesh.topology.index_map(tdim - 1)
        num_facets_local = facet_map.size_local
        assert facet_map.size_global == num_facets_local
        mesh.topology.create_connectivity(tdim - 1, tdim)
        cells = entities_to_geometry(
            mesh, tdim - 1, np.arange(num_facets_local, dtype=np.int32), False
        )
        new_x = mesh.geometry.x
        facet_type = mesh.topology.entity_types[tdim - 1]
        assert len(facet_type) == 1
        num_vertices = cell_num_vertices(facet_type[0])
        ft = facet_type[0].name
        num_vertices_global = new_x.shape[0]
        num_cells_global = cells.shape[0]
        comm.bcast((num_vertices, ft, num_vertices_global, num_cells_global), root=0)
    else:
        num_vertices, ft, num_vertices_global, num_cells_global = comm.bcast(None, root=0)
        new_x = np.empty((0, 3), dtype=np.float64)
        cells = np.empty((0, num_vertices), dtype=np.int64)

    element = ufl.Mesh(basix.ufl.element("Lagrange", ft, 1, shape=(3,)))

    if cell_type == CellType.hexahedron:
        max_facet_to_cell_links = 4
    elif cell_type == CellType.tetrahedron:
        max_facet_to_cell_links = 16
    else:
        raise ValueError("Unknown cell type")
    skeleton_mesh = create_mesh(
        comm,
        cells,
        element,
        new_x,
        partitioner(),
        ghost_mode=GhostMode.shared_facet,
        max_facet_to_cell_links=max_facet_to_cell_links,
    )

    skeleton_top = skeleton_mesh.topology
    assert (
        num_cells_global == skeleton_mesh.topology.index_map(skeleton_mesh.topology.dim).size_global
    )
    assert num_vertices_global == skeleton_mesh.topology.index_map(0).size_global

    skeleton_top.create_connectivity(1, 2)
    skeleton_f_to_c = skeleton_top.connectivity(1, 2)

    skeleton_im_f = skeleton_mesh.topology.index_map(1)

    if cell_type == CellType.hexahedron:

        def on_boundary(x):
            return np.any(np.isclose(x, 0)) or np.any(np.isclose(x, 1))

        for facet in range(skeleton_im_f.size_local):
            matched = len(skeleton_f_to_c.links(facet)) == max_facet_to_cell_links

            midpoint = compute_midpoints(skeleton_mesh, 1, np.array([facet], dtype=np.int32))[0]
            assert matched or on_boundary(midpoint)


def _round_robin_partitioner(ghost: bool):
    """Assign cell ``i`` to rank ``i % size``, optionally ghosting each
    cell on the ranks owning its dual-graph neighbours.
    """

    def partitioner(comm, nparts, dual_graph, *args):
        offset = comm.exscan(dual_graph.num_nodes) or 0
        dests, offsets = [], [0]
        for i in range(dual_graph.num_nodes):
            owner = (offset + i) % nparts
            d = [owner]
            if ghost:
                for j in dual_graph.links(i):
                    r = int(j) % nparts
                    if r not in d:
                        d.append(r)
            dests += d
            offsets.append(len(dests))
        return adjacencylist(np.array(dests, dtype=np.int32), np.array(offsets, dtype=np.int32))

    return partitioner


def _interprocess_vertices_reference(topology):
    """Global indices of the vertices attached to cells owned by two or
    more ranks, as seen by this rank.
    """
    comm = topology.comm
    c_to_v = topology.connectivity(topology.dim, 0)
    v_map = topology.index_map(0)
    vertices = {v for c in range(topology.index_map(1).size_local) for v in c_to_v.links(c)}
    attached = set(v_map.local_to_global(np.fromiter(vertices, dtype=np.int32)))
    shared = comm.allgather(attached)
    return {v for v in attached if sum(v in s for s in shared) > 1}


def _star_mesh_data(comm, num_branches):
    """Cells/geometry/element for a star of ``num_branches`` intervals
    joining at vertex 0. Geometry is arbitrary; only the topology matters.
    """
    if comm.rank == 0:
        x = np.arange(num_branches + 1, dtype=np.float64).reshape(-1, 1)
        cells = np.array([[0, i + 1] for i in range(num_branches)], dtype=np.int64)
    else:
        x = np.empty((0, 1), dtype=np.float64)
        cells = np.empty((0, 2), dtype=np.int64)
    e = ufl.Mesh(basix.ufl.element("Lagrange", "interval", 1, shape=(1,)))
    return cells, x, e


@pytest.mark.parametrize("num_branches", [2, 3, 5, 7])
@pytest.mark.parametrize("ghost", [False, True])
def test_star_interprocess_facets(num_branches, ghost):
    """A topological star of intervals joining at one vertex, distributed one
    cell per rank in turn. The joining vertex must be an interprocess facet, and
    the interval tips must be exterior facets.

    Note that because every cell is the dual graph neighbour of every other cell,
    v0 is ghosted to every rank that owns at least one branch.
    """
    comm = MPI.COMM_WORLD
    cells, x, e = _star_mesh_data(comm, num_branches)
    mesh = create_mesh(
        comm,
        cells,
        e,
        x,
        _round_robin_partitioner(ghost),
        max_facet_to_cell_links=num_branches,
    )

    topology = mesh.topology
    topology.create_connectivity(0, 1)
    v_map = topology.index_map(0)

    interprocess = set(v_map.local_to_global(topology.interprocess_facets()))
    reference = _interprocess_vertices_reference(topology)
    num_exterior = comm.allreduce(len(exterior_facet_indices(topology)), MPI.SUM)

    assert interprocess == reference
    assert num_exterior == num_branches

    # A rank beyond the number of branches does not own a branch and is not a
    # dual-graph neighbour of another branch, so it receives no cells or
    # vertices.
    if comm.rank >= num_branches:
        assert topology.index_map(1).size_local == 0
        assert v_map.size_local == 0


@pytest.mark.parametrize("gpart", _graph_partitioners)
@pytest.mark.parametrize("num_branches", [7, 11])
@pytest.mark.parametrize("ghost_mode", [GhostMode.none, GhostMode.shared_facet])
def test_star_interprocess_facets_builtin_partitioner(gpart, num_branches, ghost_mode):
    """Same star-of-intervals check as test_star_interprocess_facets, but with
    the built-in mesh partitioners.
    """
    comm = MPI.COMM_WORLD
    cells, x, e = _star_mesh_data(comm, num_branches)
    mesh = create_mesh(
        comm,
        cells,
        e,
        x,
        gpart,
        ghost_mode=ghost_mode,
        max_facet_to_cell_links=num_branches,
    )

    topology = mesh.topology
    topology.create_connectivity(0, 1)
    v_map = topology.index_map(0)

    interprocess = set(v_map.local_to_global(topology.interprocess_facets()))
    reference = _interprocess_vertices_reference(topology)
    num_exterior = comm.allreduce(len(exterior_facet_indices(topology)), MPI.SUM)
    num_owning_ranks = comm.allreduce(int(topology.index_map(1).size_local > 0), MPI.SUM)
    # num_branches must be large enough that the partitioner actually splits
    # the star across ranks - SCOTCH seems to produce empty ranks, for example.
    assert comm.size == 1 or num_owning_ranks > 1

    assert interprocess == reference
    assert num_exterior == num_branches
