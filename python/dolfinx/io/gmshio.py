# Copyright (C) 2022 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tools to extract data from Gmsh models"""
import typing

import numpy as np
import numpy.typing as npt

import basix
import basix.ufl_wrapper
import ufl
from dolfinx import cpp as _cpp
from dolfinx.cpp.graph import AdjacencyList_int32
from dolfinx.mesh import (CellType, GhostMode, Mesh, create_cell_partitioner,
                          create_mesh, meshtags, meshtags_from_entities)

from mpi4py import MPI as _MPI

__all__ = ["cell_perm_array", "ufl_mesh"]

try:
    import gmsh
    _has_gmsh = True
except ModuleNotFoundError:
    _has_gmsh = False


def ufl_mesh(gmsh_cell: int, gdim: int) -> ufl.Mesh:
    """Create a UFL mesh from a Gmsh cell identifier and the geometric dimension.

    See: http://gmsh.info//doc/texinfo/gmsh.html#MSH-file-format.

    Args:
        gmsh_cell: The Gmsh cell identifier
        gdim: The geometric dimension of the mesh

    Returns:
        A ufl Mesh using Lagrange elements (equispaced) of the
        corresponding DOLFINx cell
    """
    shape, degree = _gmsh_to_cells[gmsh_cell]
    cell = ufl.Cell(shape, geometric_dimension=gdim)

    element = basix.ufl_wrapper.create_vector_element(
        basix.ElementFamily.P, cell.cellname(), degree, basix.LagrangeVariant.equispaced, dim=gdim, gdim=gdim)
    return ufl.Mesh(element)


def cell_perm_array(cell_type: CellType, num_nodes: int) -> typing.List[int]:
    """The permutation array for permuting Gmsh ordering to DOLFINx ordering.

    Args:
        cell_type: The DOLFINx cell type
        num_nodes: The number of nodes in the cell

    Returns:
        An array `p` such that `a_dolfinx[i] = a_gmsh[p[i]]`.
    """
    return _cpp.io.perm_gmsh(cell_type, num_nodes)


if _has_gmsh:
    __all__ += ["extract_topology_and_markers", "extract_geometry", "model_to_mesh", "read_from_msh"]

    def extract_topology_and_markers(model: gmsh.model, name: str = None):
        """Extract all entities tagged with a physical marker in the gmsh
        model, and collect the data per cell type.

        Returns a nested dictionary where the first key is the gmsh MSH
        element type integer. Each element type present in the model
        contains the cell topology of the elements and corresponding
        markers.

        Args:
            model: The Gmsh model
            name: The name of the gmsh model. If not set the current
                model will be used.

        Returns:
            A nested dictionary where each key corresponds to a gmsh
            cell type. Each cell type found in the mesh has a 2D array
            containing the topology of the marked cell and a list with
            the corresponding markers.

        """
        if name is not None:
            model.setCurrent(name)

        # Get the physical groups from gmsh in the form [(dim1, tag1),(dim1,
        # tag2), (dim2, tag3),...]
        phys_grps = model.getPhysicalGroups()
        topologies: typing.Dict[int, typing.Dict[str, npt.NDArray[typing.Any]]] = {}
        for dim, tag in phys_grps:
            # Get the entities of dimension `dim`, dim=0 -> Points,
            # dim=1 - >Lines, dim=2 -> Triangles/Quadrilaterals, etc.
            entities = model.getEntitiesForPhysicalGroup(dim, tag)
            for entity in entities:
                # Get cell type, list of cells with given tag and
                # topology of tagged cells
                # NOTE: Assumes that each entity only have one
                # cell-type, i.e. facets of prisms and pyramid meshes
                # are not supported
                (entity_types, entity_tags, entity_topologies) = model.mesh.getElements(dim, tag=entity)
                assert len(entity_types) == 1

                # Determine number of local nodes per element to create the
                # topology of the elements
                properties = model.mesh.getElementProperties(entity_types[0])
                name, dim, _, num_nodes, _, _ = properties

                # Array of shape (num_elements,num_nodes_per_element)
                # containing the topology of the elements on this
                # entity.
                # NOTE: Gmsh indexing starts with one, we therefore
                # subtract 1 from each node to use zero-based numbering
                topology = entity_topologies[0].reshape(-1, num_nodes) - 1

                # Create marker array of length of number of tagged cells
                marker = np.full_like(entity_tags[0], tag)

                # Group element topology and markers of the same entity type
                entity_type = entity_types[0]
                if entity_type in topologies.keys():
                    topologies[entity_type]["topology"] = np.concatenate(
                        (topologies[entity_type]["topology"], topology), axis=0)
                    topologies[entity_type]["cell_data"] = np.hstack([topologies[entity_type]["cell_data"], marker])
                else:
                    topologies[entity_type] = {"topology": topology, "cell_data": marker}

        return topologies

    def extract_geometry(model: gmsh.model, name: str = None) -> npt.NDArray[np.float64]:
        """Extract the mesh geometry from a gmsh model as an array of shape
        (num_nodes, 3), where the i-th row corresponds to the i-th node in the
        mesh.

        Args:
            model: The Gmsh model
            name: The name of the gmsh model. If not set the current model will be used.

        Returns:
            The mesh geometry as an array of shape (num_nodes, 3).

        """
        if name is not None:
            model.setCurrent(name)

        # Get the unique tag and coordinates for nodes
        # in mesh
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

    def model_to_mesh(model: gmsh.model, comm: _MPI.Comm, rank: int,
                      gdim: int = 3,
                      partitioner: typing.Callable[
            [_MPI.Comm, int, int, AdjacencyList_int32], AdjacencyList_int32] =
            create_cell_partitioner(GhostMode.none)) -> typing.Tuple[
            Mesh, _cpp.mesh.MeshTags_int32, _cpp.mesh.MeshTags_int32]:
        """Given a Gmsh model, take all physical entities of the highest
        topological dimension and create the corresponding DOLFINx mesh.

        It is assumed that the gmsh model lives on a single rank, and is
        then read into DOLFINx on a single process to be distributed.
        This means that this function should only be called once for
        large problems. It is recommended to save the mesh and
        corresponding tags to XDMFFile after creation for efficient
        access.

        Args:
            comm: The MPI communicator to use for mesh creation
            rank: The rank the Gmsh model is initialized on
            model: The Gmsh model
            gdim: Geometrical dimension of the mesh
            partitioner: Function that computes the parallel
                distribution of cells across MPI ranks

        Returns:
            A triplet (mesh, cell_tags, facet_tags) where cell_tags hold
            markers for the cells, facet tags holds markers for facets
            if found in Gmsh model.
        """

        if comm.rank == rank:
            # Get mesh geometry
            x = extract_geometry(model)

            # Get mesh topology for each element
            topologies = extract_topology_and_markers(model)

            # Extract Gmsh cell id, dimension of cell and number of
            # nodes to cell for each
            num_cell_types = len(topologies.keys())
            cell_information = {}
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

            # Check for facet data and broadcast relevant info if True
            has_facet_data = False
            if tdim - 1 in cell_dimensions:
                has_facet_data = True

            has_facet_data = comm.bcast(has_facet_data, root=rank)
            if has_facet_data:
                num_facet_nodes = comm.bcast(cell_information[perm_sort[-2]]["num_nodes"], root=rank)
                gmsh_facet_id = cell_information[perm_sort[-2]]["id"]
                marked_facets = np.asarray(topologies[gmsh_facet_id]["topology"], dtype=np.int64)
                facet_values = np.asarray(topologies[gmsh_facet_id]["cell_data"], dtype=np.int32)

            cells = np.asarray(topologies[cell_id]["topology"], dtype=np.int64)
            cell_values = np.asarray(topologies[cell_id]["cell_data"], dtype=np.int32)

        else:
            cell_id, num_nodes = comm.bcast([None, None], root=rank)
            cells, x = np.empty([0, num_nodes], dtype=np.int32), np.empty([0, gdim])
            cell_values = np.empty((0,), dtype=np.int32)
            has_facet_data = comm.bcast(None, root=rank)
            if has_facet_data:
                num_facet_nodes = comm.bcast(None, root=rank)
                marked_facets = np.empty((0, num_facet_nodes), dtype=np.int32)
                facet_values = np.empty((0,), dtype=np.int32)

        # Create distributed mesh
        ufl_domain = ufl_mesh(cell_id, gdim)
        gmsh_cell_perm = cell_perm_array(_cpp.mesh.to_type(str(ufl_domain.ufl_cell())), num_nodes)
        cells = cells[:, gmsh_cell_perm]
        mesh = create_mesh(comm, cells, x[:, :gdim], ufl_domain, partitioner)

        # Create MeshTags for cells
        local_entities, local_values = _cpp.io.distribute_entity_data(mesh, mesh.topology.dim, cells, cell_values)
        mesh.topology.create_connectivity(mesh.topology.dim, 0)
        adj = _cpp.graph.AdjacencyList_int32(local_entities)
        ct = meshtags_from_entities(mesh, mesh.topology.dim, adj, local_values.astype(np.int32))
        ct.name = "Cell tags"

        # Create MeshTags for facets
        topology = mesh.topology
        if has_facet_data:
            # Permute facets from MSH to Dolfin-X ordering
            # FIXME: This does not work for prism meshes
            if topology.cell_type == CellType.prism or topology.cell_type == CellType.pyramid:
                raise RuntimeError(f"Unsupported cell type {topology.cell_type}")

            facet_type = _cpp.mesh.cell_entity_type(_cpp.mesh.to_type(str(ufl_domain.ufl_cell())), topology.dim - 1, 0)
            gmsh_facet_perm = cell_perm_array(facet_type, num_facet_nodes)
            marked_facets = marked_facets[:, gmsh_facet_perm]

            local_entities, local_values = _cpp.io.distribute_entity_data(
                mesh, mesh.topology.dim - 1, marked_facets, facet_values)
            mesh.topology.create_connectivity(topology.dim - 1, topology.dim)
            adj = _cpp.graph.AdjacencyList_int32(local_entities)
            ft = meshtags_from_entities(mesh, topology.dim - 1, adj, local_values.astype(np.int32))
            ft.name = "Facet tags"
        else:
            ft = meshtags(mesh, topology.dim - 1, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32))

        return (mesh, ct, ft)

    def read_from_msh(
        filename: str, comm: _MPI.Comm, rank: int = 0,
        gdim: int = 3,
        partitioner: typing.Callable[
            [_MPI.Comm, int, int, AdjacencyList_int32], AdjacencyList_int32] =
            create_cell_partitioner(GhostMode.none)) -> typing.Tuple[
                Mesh, _cpp.mesh.MeshTags_int32, _cpp.mesh.MeshTags_int32]:
        """Reads a mesh from a msh-file and returns the distributed DOLFINx
        mesh and cell and facet markers associated with physical groups
        in the msh file.

        Args:
            filename: Name of msh file
            comm: The MPI communicator to initialize the mesh with
            rank: Rank for `comm` responsible for reading the msh file
            gdim: Geometrical dimension of the mesh

        Returns:
            A triplet (mesh, cell_tags, facet_tags) with meshtags for
            associated physical groups for cells and facets

        """
        if comm.rank == rank:
            gmsh.initialize()
            gmsh.model.add("Mesh from file")
            gmsh.merge(filename)

        output = model_to_mesh(gmsh.model, comm, rank, gdim=gdim, partitioner=partitioner)

        if comm.rank == rank:
            gmsh.finalize()
        return output

    # Map from Gmsh cell type identifier (integer) to DOLFINx cell type
    # and degree http://gmsh.info//doc/texinfo/gmsh.html#MSH-file-format
    _gmsh_to_cells = {1: ("interval", 1), 2: ("triangle", 1),
                      3: ("quadrilateral", 1), 4: ("tetrahedron", 1),
                      5: ("hexahedron", 1), 8: ("interval", 2),
                      9: ("triangle", 2), 10: ("quadrilateral", 2),
                      11: ("tetrahedron", 2), 12: ("hexahedron", 2),
                      15: ("point", 0), 21: ("triangle", 3),
                      26: ("interval", 3), 29: ("tetrahedron", 3),
                      36: ("quadrilateral", 3)}
