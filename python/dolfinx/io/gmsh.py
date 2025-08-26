# Copyright (C) 2022-2025 Jørgen S. Dokken, Henrik N. T. Finsberg and
# Paul T. Kühner
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tools to extract data from Gmsh models."""

import typing
from collections.abc import Callable
from pathlib import Path

from mpi4py import MPI as _MPI

import numpy as np
import numpy.typing as npt

import basix
import basix.ufl
import ufl
from dolfinx import cpp as _cpp
from dolfinx import default_real_type
from dolfinx.cpp.graph import AdjacencyList_int32 as _AdjacencyList_int32
from dolfinx.graph import AdjacencyList, adjacencylist
from dolfinx.io.utils import distribute_entity_data
from dolfinx.mesh import CellType, Mesh, MeshTags, create_mesh, meshtags_from_entities

__all__ = [
    "cell_perm_array",
    "extract_geometry",
    "extract_topology_and_markers",
    "model_to_mesh",
    "read_from_msh",
    "ufl_mesh",
]


class TopologyDict(typing.TypedDict):
    """TopologyDict is a TypedDict for storing the topology of the marked
    cell.

    Args:
        topology: 2D array containing the topology of the marked cell.
        cell_data: List with the corresponding markers.

    Note:
        The TypedDict is only used for type hinting, and does not
        enforce the structure of the dictionary, but rather provides
        a hint to the user and the type checker.
    """

    topology: npt.NDArray[typing.Any]
    cell_data: npt.NDArray[typing.Any]


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


class PhysicalGroup(typing.NamedTuple):
    """Physical group info.

    Args:
        dim: dimension of the physical group
        tag: tag of the physical group
    """

    dim: int
    tag: int


class MeshData(typing.NamedTuple):
    """Data for representing a mesh and associated tags.

    Args:
        mesh: Mesh.
        cell_tags: MeshTags for cells.
        facet_tags: MeshTags for facets (codim 1).
        ridge_tags: MeshTags for ridges (codim 2).
        peak_tags: MeshTags for peaks (codim 3).
        physical_groups: Physical groups in the mesh, where the key
            is the physical name and the value is a tuple with the
            dimension and tag.
    """

    mesh: Mesh
    cell_tags: typing.Optional[MeshTags]
    facet_tags: typing.Optional[MeshTags]
    ridge_tags: typing.Optional[MeshTags]
    peak_tags: typing.Optional[MeshTags]
    physical_groups: dict[str, PhysicalGroup]


def ufl_mesh(gmsh_cell: int, gdim: int, dtype: npt.DTypeLike) -> ufl.Mesh:
    """Create a UFL mesh from a Gmsh cell identifier and geometric
    dimension.

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
    """The permutation array for permuting Gmsh ordering to DOLFINx
    ordering.

    Args:
        cell_type: DOLFINx cell type.
        num_nodes: Number of nodes in the cell.

    Returns:
        An array ``p`` such that ``a_dolfinx[i] = a_gmsh[p[i]]``.

    """
    return _cpp.io.perm_gmsh(cell_type, num_nodes)


def extract_topology_and_markers(
    model, name: typing.Optional[str] = None
) -> tuple[dict[int, TopologyDict], dict[str, PhysicalGroup]]:
    """Extract all entities tagged with a physical marker in the gmsh
    model.

    Returns a nested dictionary where the first key is the gmsh MSH
    element type integer. Each element type present in the model
    contains the cell topology of the elements and corresponding
    markers.

    Args:
        model: Gmsh model.
        name: Name of the gmsh model. If not set the current
            model will be used.

    Returns:
        A tuple ``(topologies, physical_groups)``, where ``topologies`` is
        a nested dictionary where each key corresponds to a gmsh cell type.
        Each cell type found in the mesh has a 2D array containing the
        topology of the marked cell and a list with the corresponding
        markers. ``physical_groups`` is a dictionary where the key is the
        physical name and the value is a tuple with the dimension and tag.

    """
    if name is not None:
        model.setCurrent(name)

    # Get the physical groups from gmsh in the form [(dim1, tag1),
    # (dim1, tag2), (dim2, tag3),...]
    phys_grps = model.getPhysicalGroups()
    topologies: dict[int, TopologyDict] = {}
    # Create a dictionary with the physical groups where the key is the
    # physical name and the value is a tuple with the dimension and tag
    physical_groups: dict[str, PhysicalGroup] = {}
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

            if len(entity_types) > 1:
                raise RuntimeError(
                    f"Unsupported mesh with multiple cell types {entity_types} for entity {entity}"
                )
            elif len(entity_types) == 0:
                continue

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

        physical_groups[model.getPhysicalName(dim, tag)] = PhysicalGroup(dim, tag)

    return topologies, physical_groups


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
        Callable[[_MPI.Comm, int, int, _AdjacencyList_int32], _AdjacencyList_int32]
    ] = None,
    dtype=default_real_type,
) -> MeshData:
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
        MeshData with mesh and tags of corresponding entities by
        codimension. Codimension 0 is the cell tags, codimension 1 is the
        facet tags, codimension 2 is the ridge tags and codimension 3 is
        the peak tags as well as a lookup table from the physical groups by
        name to integer.

    Note:
        For performance, this function should only be called once for
        large problems. For reuse, it is recommended to save the mesh
        and corresponding tags using :class:`dolfinx.io.XDMFFile` after
        creation for efficient access.
    """
    if comm.rank == rank:
        assert model is not None, "Gmsh model is None on rank responsible for mesh creation."
        # Get mesh geometry and mesh topology for each element
        x = extract_geometry(model)
        topologies, physical_groups = extract_topology_and_markers(model)

        if len(physical_groups) == 0:
            raise RuntimeError("No 'physical groups' in gmsh mesh. Cannot continue.")

        # Extract Gmsh entity (cell) id, topological dimension and number
        # of nodes which is used to create an appropriate coordinate
        # element, and seperate higher topological entities from lower
        # topological entities (e.g. facets, ridges and peaks).
        num_unique_entities = len(topologies.keys())
        element_ids = np.zeros(num_unique_entities, dtype=np.int32)
        entity_tdim = np.zeros(num_unique_entities, dtype=np.int32)
        num_nodes_per_element = np.zeros(num_unique_entities, dtype=np.int32)
        for i, element in enumerate(topologies.keys()):
            _, dim, _, num_nodes, _, _ = model.mesh.getElementProperties(element)
            element_ids[i] = element
            entity_tdim[i] = dim
            num_nodes_per_element[i] = num_nodes

        # Broadcast information to all other ranks
        entity_tdim, element_ids, num_nodes_per_element = comm.bcast(
            (entity_tdim, element_ids, num_nodes_per_element), root=rank
        )
    else:
        entity_tdim, element_ids, num_nodes_per_element = comm.bcast((None, None, None), root=rank)

    # Sort elements by descending dimension
    assert len(np.unique(entity_tdim)) == len(entity_tdim)
    perm_sort = np.argsort(entity_tdim)[::-1]

    # Extract position of the highest topological entity and its
    # topological dimension
    cell_position = perm_sort[0]
    tdim = int(entity_tdim[cell_position])

    # Extract entity -> node connectivity for all cells and sub-entities
    # marked in the GMSH model
    meshtags: dict[int, tuple[npt.NDArray[np.int64], npt.NDArray[np.int32]]] = {}
    for position in perm_sort:
        codim = tdim - entity_tdim[position]
        if comm.rank == rank:
            gmsh_entity_id = element_ids[position]
            marked_entities = np.asarray(topologies[gmsh_entity_id]["topology"], dtype=np.int64)
            entity_values = np.asarray(topologies[gmsh_entity_id]["cell_data"], dtype=np.int32)
            meshtags[codim] = (marked_entities, entity_values)
        else:
            # Any other process than input rank does not have any entities
            marked_entities = np.empty((0, num_nodes_per_element[position]), dtype=np.int32)
            entity_values = np.empty((0,), dtype=np.int32)
            meshtags[codim] = (marked_entities, entity_values)

    # Create a UFL Mesh object for the GMSH element with the highest
    # topoligcal dimension
    ufl_domain = ufl_mesh(element_ids[cell_position], gdim, dtype=dtype)

    # Get cell->node connectivity and  permute to FEniCS ordering
    num_nodes = num_nodes_per_element[cell_position]
    gmsh_cell_perm = cell_perm_array(_cpp.mesh.to_type(str(ufl_domain.ufl_cell())), num_nodes)
    cell_connectivity = meshtags[0][0][:, gmsh_cell_perm].copy()

    # Create a distributed mesh, where mesh nodes are only destributed from
    # the input rank
    if comm.rank != rank:
        x = np.empty([0, gdim], dtype=dtype)  # No nodes on other than root rank
    mesh = create_mesh(
        comm, cell_connectivity, ufl_domain, x[:, :gdim].astype(dtype, copy=False), partitioner
    )
    assert tdim == mesh.topology.dim, (
        f"{mesh.topology.dim=} does not match Gmsh model dimension {tdim}"
    )

    # Create MeshTags for all sub entities
    topology = mesh.topology
    codim_to_name = {0: "cell", 1: "facet", 2: "ridge", 3: "peak"}
    dolfinx_meshtags: dict[str, typing.Optional[MeshTags]] = {}
    for codim in [0, 1, 2, 3]:
        key = f"{codim_to_name[codim]}_tags"
        if (
            codim == 1 and topology.cell_type == CellType.prism
        ) or topology.cell_type == CellType.pyramid:
            raise RuntimeError(f"Unsupported facet tag for type {topology.cell_type}")

        meshtag_data = meshtags.get(codim, None)
        if meshtag_data is None:
            dolfinx_meshtags[key] = None
            continue

        # Distribute entity data [[e0_v0, e0_v1, ...], [e1_v0, e1_v1, ...],
        # ...] which is made in global input indices to local indices on
        # the owning process
        (marked_entities, entity_values) = meshtag_data
        local_entities, local_values = distribute_entity_data(
            mesh, tdim - codim, marked_entities, entity_values
        )
        # Create MeshTags object from the local entities
        mesh.topology.create_connectivity(tdim - codim, tdim)
        adj = adjacencylist(local_entities)
        et = meshtags_from_entities(
            mesh, tdim - codim, adj, local_values.astype(np.int32, copy=False)
        )
        et.name = key
        dolfinx_meshtags[key] = et

    # Broadcast physical groups (string to integer mapping) to all ranks
    if comm.rank == rank:
        physical_groups = comm.bcast(physical_groups, root=rank)
    else:
        physical_groups = comm.bcast(None, root=rank)

    return MeshData(mesh, physical_groups=physical_groups, **dolfinx_meshtags)


def read_from_msh(
    filename: typing.Union[str, Path],
    comm: _MPI.Comm,
    rank: int = 0,
    gdim: int = 3,
    partitioner: typing.Optional[
        Callable[[_MPI.Comm, int, int, AdjacencyList], _AdjacencyList_int32]
    ] = None,
) -> MeshData:
    """Read a Gmsh .msh file and return a :class:`dolfinx.mesh.Mesh` and
    cell facet markers.

    Note:
        This function requires the Gmsh Python module.

    Args:
        filename: Name of ``.msh`` file.
        comm: MPI communicator to create the mesh on.
        rank: Rank of ``comm`` responsible for reading the ``.msh``
            file.
        gdim: Geometric dimension of the mesh

    Returns:
        Meshdata with mesh, cell tags, facet tags, edge tags,
        vertex tags and physical groups.

    """
    try:
        import gmsh
    except ModuleNotFoundError:
        # Python 3.11+ adds the add_note method to exceptions
        # e.add_note("Gmsh must be installed to import dolfinx.io.gmsh")
        raise ModuleNotFoundError(
            "No module named 'gmsh': dolfinx.io.gmsh.read_from_msh requires Gmsh.", name="gmsh"
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
