# Copyright (C) 2022 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tools to extract data from Gmsh models."""

import typing
from pathlib import Path

from mpi4py import MPI as _MPI

import numpy as np
import numpy.typing as npt

import basix
import basix.ufl
import ufl
from dolfinx import cpp as _cpp
from dolfinx import default_real_type
from dolfinx.cpp.graph import AdjacencyList_int32
from dolfinx.graph import AdjacencyList, adjacencylist
from dolfinx.io.utils import distribute_entity_data
from dolfinx.mesh import CellType, Mesh, create_mesh, meshtags, meshtags_from_entities, MeshTags

__all__ = [
    "cell_perm_array",
    "extract_geometry",
    "extract_topology_and_markers",
    "model_to_mesh",
    "read_from_msh",
    "ufl_mesh",
]


# Map from Gmsh cell type identifier (integer) to DOLFINx cell type and
# degree https://gmsh.info//doc/texinfo/gmsh.html#MSH-file-format
_gmsh_to_cells = {
    1: ("interval", 1),
    2: ("triangle", 1),
    3: ("quadrilateral", 1),
    4: ("tetrahedron", 1),
    5: ("hexahedron", 1),
    8: ("interval", 2),
    9: ("triangle", 2),
    10: ("quadrilateral", 2),
    11: ("tetrahedron", 2),
    12: ("hexahedron", 2),
    15: ("point", 0),
    21: ("triangle", 3),
    26: ("interval", 3),
    29: ("tetrahedron", 3),
    36: ("quadrilateral", 3),
    92: ("hexahedron", 3),
}


class GMSHModel(typing.NamedTuple):
    mesh: Mesh
    cell_tags: MeshTags
    facet_tags: MeshTags
    edge_tags: MeshTags
    vertex_tags: MeshTags


def ufl_mesh(gmsh_cell: int, gdim: int, dtype: npt.DTypeLike) -> ufl.Mesh:
    """Create a UFL mesh from a Gmsh cell identifier and geometric dimension.

    See https://gmsh.info//doc/texinfo/gmsh.html#MSH-file-format.

    Args:
        gmsh_cell: Gmsh cell identifier.
        gdim: Geometric dimension of the mesh.

    Returns:
        UFL Mesh using Lagrange elements (equispaced) of the
        corresponding DOLFINx cell.

    """
    try:
        shape, degree = _gmsh_to_cells[gmsh_cell]
    except KeyError as e:
        print(f"Unknown cell type {gmsh_cell}.")
        raise e
    cell = ufl.Cell(shape)
    element = basix.ufl.element(
        basix.ElementFamily.P,
        cell.cellname(),
        degree,
        basix.LagrangeVariant.equispaced,
        shape=(gdim,),
        dtype=dtype,  # type: ignore[arg-type]
    )
    return ufl.Mesh(element)


def cell_perm_array(cell_type: CellType, num_nodes: int) -> list[int]:
    """The permutation array for permuting Gmsh ordering to DOLFINx ordering.

    Args:
        cell_type: DOLFINx cell type.
        num_nodes: Number of nodes in the cell.

    Returns:
        An array ``p`` such that ``a_dolfinx[i] = a_gmsh[p[i]]``.

    """
    return _cpp.io.perm_gmsh(cell_type, num_nodes)


def extract_topology_and_markers(model, name: typing.Optional[str] = None):
    """Extract all entities tagged with a physical marker in the gmsh model.

    Returns a nested dictionary where the first key is the gmsh MSH
    element type integer. Each element type present in the model
    contains the cell topology of the elements and corresponding
    markers.

    Args:
        model: Gmsh model.
        name: Name of the gmsh model. If not set the current
            model will be used.

    Returns:
        A nested dictionary where each key corresponds to a gmsh cell
        type. Each cell type found in the mesh has a 2D array containing
        the topology of the marked cell and a list with the
        corresponding markers.

    """
    if name is not None:
        model.setCurrent(name)

    # Get the physical groups from gmsh in the form [(dim1, tag1),
    # (dim1, tag2), (dim2, tag3),...]
    phys_grps = model.getPhysicalGroups()
    topologies: dict[int, dict[str, npt.NDArray[typing.Any]]] = {}
    for dim, tag in phys_grps:
        # Get the entities of dimension `dim`, dim=0 -> Points, dim=1 -
        # >Lines, dim=2 -> Triangles/Quadrilaterals, etc.
        entities = model.getEntitiesForPhysicalGroup(dim, tag)
        for entity in entities:
            # Get cell type, list of cells with given tag and topology
            # of tagged cells NOTE: Assumes that each entity only have
            # one cell-type,
            # i.e. facets of prisms and pyramid meshes are not supported
            (entity_types, entity_tags, entity_topologies) = model.mesh.getElements(dim, tag=entity)
            assert len(entity_types) == 1

            # Determine number of local nodes per element to create the
            # topology of the elements
            properties = model.mesh.getElementProperties(entity_types[0])
            name, dim, _, num_nodes, _, _ = properties

            # Array of shape (num_elements,num_nodes_per_element)
            # containing the topology of the elements on this entity.
            # NOTE: Gmsh indexing starts with one, we therefore subtract
            # 1 from each node to use zero-based numbering
            topology = entity_topologies[0].reshape(-1, num_nodes) - 1

            # Create marker array of length of number of tagged cells
            marker = np.full_like(entity_tags[0], tag)

            # Group element topology and markers of the same entity type
            entity_type = entity_types[0]
            if entity_type in topologies.keys():
                topologies[entity_type]["topology"] = np.concatenate(
                    (topologies[entity_type]["topology"], topology), axis=0
                )
                topologies[entity_type]["cell_data"] = np.hstack(
                    [topologies[entity_type]["cell_data"], marker]
                )
            else:
                topologies[entity_type] = {"topology": topology, "cell_data": marker}

    return topologies


def extract_geometry(model, name: typing.Optional[str] = None) -> npt.NDArray[np.float64]:
    """Extract the mesh geometry from a Gmsh model.

    Returns an array of shape ``(num_nodes, 3)``, where the i-th row
    corresponds to the i-th node in the mesh.

    Args:
        model: Gmsh model
        name: Name of the Gmsh model. If not set the current
            model will be used.

    Returns:
        The mesh geometry as an array of shape ``(num_nodes, 3)``.

    """
    if name is not None:
        model.setCurrent(name)

    # Get the unique tag and coordinates for nodes in mesh
    indices, points, _ = model.mesh.getNodes()
    points = points.reshape(-1, 3)

    # Gmsh indices starts at 1. We therefore subtract one to use
    # zero-based numbering
    indices -= 1

    # In some cases, Gmsh does not return the points in the same
    # order as their unique node index. We therefore sort nodes in
    # geometry according to the unique index
    perm_sort = np.argsort(indices)
    assert np.all(indices[perm_sort] == np.arange(len(indices)))
    return points[perm_sort]


def model_to_mesh(
    model,
    comm: _MPI.Comm,
    rank: int,
    gdim: int = 3,
    partitioner: typing.Optional[
        typing.Callable[[_MPI.Comm, int, int, AdjacencyList_int32], AdjacencyList_int32]
    ] = None,
    dtype=default_real_type,
) -> GMSHModel:
    """Create a Mesh from a Gmsh model.

    Creates a :class:`dolfinx.mesh.Mesh` from the physical entities of
    the highest topological dimension in the Gmsh model. In parallel,
    the gmsh model is processed on one MPI rank, and the
    :class:`dolfinx.mesh.Mesh` is distributed across ranks.

    Args:
        model: Gmsh model.
        comm: MPI communicator to use for mesh creation.
        rank: MPI rank that the Gmsh model is initialized on.
        gdim: Geometrical dimension of the mesh.
        partitioner: Function that computes the parallel
            distribution of cells across MPI ranks.

    Returns:
        A NamedTuple ``(mesh, cell_tags, facet_tags, edge_tags, vertex_tags)``,
        where cell_tags hold markers for the cells, facet tags holds markers
        for facets (if tags are found in Gmsh model), edge_tags holds markers
        for edges (if tags are found in Gmsh model), and vertex_tags holds
        markers for vertices (if tags are found in Gmsh model).

    Note:
        For performance, this function should only be called once for
        large problems. For re-use, it is recommended to save the mesh
        and corresponding tags using :class:`dolfinx.io.XDMFFile` after
        creation for efficient access.

    """
    if comm.rank == rank:
        assert model is not None, "Gmsh model is None on rank responsible for mesh creation."
        # Get mesh geometry and mesh topology for each element
        x = extract_geometry(model)
        topologies = extract_topology_and_markers(model)

        # Extract Gmsh cell id, dimension of cell and number of nodes to
        # cell for each
        num_cell_types = len(topologies.keys())
        cell_information = dict()
        cell_dimensions = np.zeros(num_cell_types, dtype=np.int32)
        for i, element in enumerate(topologies.keys()):
            _, dim, _, num_nodes, _, _ = model.mesh.getElementProperties(element)
            cell_information[i] = {"id": element, "dim": dim, "num_nodes": num_nodes}
            cell_dimensions[i] = dim

        # Sort elements by ascending dimension
        perm_sort = np.argsort(cell_dimensions)

        # Broadcast cell type data and geometric dimension
        cell_id = cell_information[perm_sort[-1]]["id"]
        tdim = cell_information[perm_sort[-1]]["dim"]
        num_nodes = cell_information[perm_sort[-1]]["num_nodes"]
        cell_id, num_nodes = comm.bcast([cell_id, num_nodes], root=rank)

        # Check for facet, edge and vertex data and broadcast relevant info if True
        has_facet_data = False
        if tdim - 1 in cell_dimensions:
            has_facet_data = True
        has_edge_data = False
        if tdim - 2 in cell_dimensions:
            has_edge_data = True
        has_vertex_data = False
        if tdim - 3 in cell_dimensions:
            has_vertex_data = True

        has_facet_data = comm.bcast(has_facet_data, root=rank)
        if has_facet_data:
            num_facet_nodes = comm.bcast(cell_information[perm_sort[-2]]["num_nodes"], root=rank)
            gmsh_facet_id = cell_information[perm_sort[-2]]["id"]
            marked_facets = np.asarray(topologies[gmsh_facet_id]["topology"], dtype=np.int64)
            facet_values = np.asarray(topologies[gmsh_facet_id]["cell_data"], dtype=np.int32)

        has_edge_data = comm.bcast(has_edge_data, root=rank)
        if has_edge_data:
            num_edge_nodes = comm.bcast(cell_information[perm_sort[-3]]["num_nodes"], root=rank)
            gmsh_edge_id = cell_information[perm_sort[-3]]["id"]
            marked_edges = np.asarray(topologies[gmsh_edge_id]["topology"], dtype=np.int64)
            edge_values = np.asarray(topologies[gmsh_edge_id]["cell_data"], dtype=np.int32)

        has_vertex_data = comm.bcast(has_vertex_data, root=rank)
        if has_vertex_data:
            num_vertex_nodes = comm.bcast(cell_information[perm_sort[-4]]["num_nodes"], root=rank)
            gmsh_vertex_id = cell_information[perm_sort[-4]]["id"]
            marked_vertices = np.asarray(topologies[gmsh_vertex_id]["topology"], dtype=np.int64)
            vertex_values = np.asarray(topologies[gmsh_vertex_id]["cell_data"], dtype=np.int32)

        cells = np.asarray(topologies[cell_id]["topology"], dtype=np.int64)
        cell_values = np.asarray(topologies[cell_id]["cell_data"], dtype=np.int32)
    else:
        cell_id, num_nodes = comm.bcast([None, None], root=rank)
        cells, x = np.empty([0, num_nodes], dtype=np.int32), np.empty([0, gdim], dtype=dtype)
        cell_values = np.empty((0,), dtype=np.int32)

        has_facet_data = comm.bcast(None, root=rank)
        if has_facet_data:
            num_facet_nodes = comm.bcast(None, root=rank)
            marked_facets = np.empty((0, num_facet_nodes), dtype=np.int32)
            facet_values = np.empty((0,), dtype=np.int32)

        has_edge_data = comm.bcast(None, root=rank)
        if has_edge_data:
            num_edge_nodes = comm.bcast(None, root=rank)
            marked_edges = np.empty((0, num_edge_nodes), dtype=np.int32)
            edge_values = np.empty((0,), dtype=np.int32)

        has_vertex_data = comm.bcast(None, root=rank)
        if has_vertex_data:
            num_vertex_nodes = comm.bcast(None, root=rank)
            marked_vertices = np.empty((0, num_vertex_nodes), dtype=np.int32)
            vertex_values = np.empty((0,), dtype=np.int32)

    # Create distributed mesh
    ufl_domain = ufl_mesh(cell_id, gdim, dtype=dtype)
    gmsh_cell_perm = cell_perm_array(_cpp.mesh.to_type(str(ufl_domain.ufl_cell())), num_nodes)
    cells = cells[:, gmsh_cell_perm].copy()
    mesh = create_mesh(comm, cells, x[:, :gdim].astype(dtype, copy=False), ufl_domain, partitioner)

    # Create MeshTags for cells
    local_entities, local_values = distribute_entity_data(
        mesh, mesh.topology.dim, cells, cell_values
    )
    mesh.topology.create_connectivity(mesh.topology.dim, 0)
    adj = adjacencylist(local_entities)
    ct = meshtags_from_entities(
        mesh, mesh.topology.dim, adj, local_values.astype(np.int32, copy=False)
    )
    ct.name = "Cell tags"

    # Create MeshTags for facets
    topology = mesh.topology
    tdim = topology.dim
    if has_facet_data:
        # Permute facets from MSH to DOLFINx ordering
        # FIXME: This does not work for prism meshes
        if topology.cell_type == CellType.prism or topology.cell_type == CellType.pyramid:
            raise RuntimeError(f"Unsupported cell type {topology.cell_type}")

        facet_type = _cpp.mesh.cell_entity_type(
            _cpp.mesh.to_type(str(ufl_domain.ufl_cell())), tdim - 1, 0
        )
        gmsh_facet_perm = cell_perm_array(facet_type, num_facet_nodes)
        marked_facets = marked_facets[:, gmsh_facet_perm]

        local_entities, local_values = distribute_entity_data(
            mesh, tdim - 1, marked_facets, facet_values
        )
        mesh.topology.create_connectivity(topology.dim - 1, tdim)
        adj = adjacencylist(local_entities)
        ft = meshtags_from_entities(mesh, tdim - 1, adj, local_values.astype(np.int32, copy=False))
        ft.name = "Facet tags"
    else:
        ft = meshtags(mesh, tdim - 1, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32))

    if has_edge_data:
        # Permute edges from MSH to DOLFINx ordering
        edge_type = _cpp.mesh.cell_entity_type(
            _cpp.mesh.to_type(str(ufl_domain.ufl_cell())), tdim - 2, 0
        )
        gmsh_edge_perm = cell_perm_array(edge_type, num_edge_nodes)
        marked_edges = marked_edges[:, gmsh_edge_perm]

        local_entities, local_values = distribute_entity_data(
            mesh, tdim - 2, marked_edges, edge_values
        )
        mesh.topology.create_connectivity(topology.dim - 2, tdim)
        adj = adjacencylist(local_entities)
        et = meshtags_from_entities(mesh, tdim - 2, adj, local_values.astype(np.int32, copy=False))
        et.name = "Edge tags"
    else:
        et = meshtags(mesh, tdim - 2, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32))

    if has_vertex_data:
        # Permute vertices from MSH to DOLFINx ordering
        vertex_type = _cpp.mesh.cell_entity_type(
            _cpp.mesh.to_type(str(ufl_domain.ufl_cell())), tdim - 3, 0
        )
        gmsh_vertex_perm = cell_perm_array(vertex_type, num_vertex_nodes)
        marked_vertices = marked_vertices[:, gmsh_vertex_perm]

        local_entities, local_values = distribute_entity_data(
            mesh, tdim - 3, marked_vertices, vertex_values
        )
        mesh.topology.create_connectivity(topology.dim - 3, tdim)
        adj = adjacencylist(local_entities)
        vt = meshtags_from_entities(mesh, tdim - 3, adj, local_values.astype(np.int32, copy=False))
        vt.name = "Vertex tags"
    else:
        vt = meshtags(mesh, tdim - 3, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32))

    return GMSHModel(mesh, ct, ft, et, vt)


def read_from_msh(
    filename: typing.Union[str, Path],
    comm: _MPI.Comm,
    rank: int = 0,
    gdim: int = 3,
    partitioner: typing.Optional[
        typing.Callable[[_MPI.Comm, int, int, AdjacencyList], AdjacencyList_int32]
    ] = None,
) -> GMSHModel:
    """Read a Gmsh .msh file and return a :class:`dolfinx.mesh.Mesh` and cell facet markers.

    Note:
        This function requires the Gmsh Python module.

    Args:
        filename: Name of ``.msh`` file.
        comm: MPI communicator to create the mesh on.
        rank: Rank of ``comm`` responsible for reading the ``.msh``
            file.
        gdim: Geometric dimension of the mesh

    Returns:
        A NamedTuple ``(mesh, cell_tags, facet_tags, edge_tags, vertex_tags)``
        with meshtags for associated physical groups for cells and facets.

    """
    try:
        import gmsh
    except ModuleNotFoundError:
        # Python 3.11+ adds the add_note method to exceptions
        # e.add_note("Gmsh must be installed to import dolfinx.io.gmshio")
        raise ModuleNotFoundError(
            "No module named 'gmsh': dolfinx.io.gmshio.read_from_msh requires Gmsh.", name="gmsh"
        )

    if comm.rank == rank:
        gmsh.initialize()
        gmsh.model.add("Mesh from file")
        gmsh.merge(str(filename))
        msh = model_to_mesh(gmsh.model, comm, rank, gdim=gdim, partitioner=partitioner)
        gmsh.finalize()
        return msh
    else:
        return model_to_mesh(gmsh.model, comm, rank, gdim=gdim, partitioner=partitioner)
