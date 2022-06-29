# Copyright (C) 2022 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tools to extract data from GMSH models"""
import typing

import numpy as np
import numpy.typing as npt
import ufl
from dolfinx import cpp as _cpp
from dolfinx.mesh import CellType

__all__ = []

try:
    import gmsh
    _has_gmsh = True
except ModuleNotFoundError:
    _has_gmsh = False

if _has_gmsh:
    __all__ += ["extract_topology_and_markers", "extract_geometry", "ufl_mesh", "cell_perm_array"]

    def extract_topology_and_markers(model: gmsh.model, name: str = None):
        """Extract all entities tagged with a physical marker in the gmsh
        model, and collect the data per cell type. Returns a nested
        dictionary where the first key is the gmsh MSH element type integer.
        Each element type present in the model contains the cell topology of
        the elements and corresponding markers.

        Args:
            model: The GMSH model
            name: The name of the gmsh model. If not set the current model will be used

        Returns:
            A nested dictionary where each key corresponds to a gmsh cell type.
            Each cell type found in the mesh has a 2D array containing the topology
            of the marked cell and a list with the corresponding markers.

        """
        if name is not None:
            model.setCurrent(name)

        # Get the physical groups from gmsh in the form [(dim1, tag1),(dim1,
        # tag2), (dim2, tag3),...]
        phys_grps = model.getPhysicalGroups()
        topologies: typing.Dict[int, typing.Dict[str, npt.NDArray[typing.Any]]] = {}
        for dim, tag in phys_grps:
            # Get the entities of dimension `dim`
            # dim=0->Points, dim=1->Lines, dim=2->Triangles/Quadrilaterals,
            # etc.
            entities = model.getEntitiesForPhysicalGroup(dim, tag)
            for entity in entities:
                # Get cell type, list of cells with given tag and topology of tagged cells
                # NOTE: Assumes that each entity only have one cell-type, i.e. facets of prisms
                # and pyramid meshes are not supported
                (entity_types, entity_tags, entity_topologies) = model.mesh.getElements(dim, tag=entity)
                assert len(entity_types) == 1

                # Determine number of local nodes per element to create the
                # topology of the elements
                properties = model.mesh.getElementProperties(entity_types[0])
                name, dim, _, num_nodes, _, _ = properties

                # Array of shape (num_elements,num_nodes_per_element)
                # containing the topology of the elements on this entity.
                # NOTE: GMSH indexing starts with one, we therefore subtract 1 from
                # each node to use zero-based numbering
                topology = entity_topologies[0].reshape(-1, num_nodes) - 1

                # Create marker array of length of number of tagged cells
                marker = np.full_like(entity_tags[0], tag)

                # Group element topology and markers of the same entity type
                entity_type = entity_types[0]
                if entity_type in topologies.keys():
                    topologies[entity_type]["topology"] = np.concatenate(
                        (topologies[entity_type]["topology"], topology), axis=0)
                    topologies[entity_type]["cell_data"] = np.hstack(
                        [topologies[entity_type]["cell_data"], marker])
                else:
                    topologies[entity_type] = {"topology": topology,
                                               "cell_data": marker}

        return topologies

    def extract_geometry(model: gmsh.model, name: str = None):
        """Extract the mesh geometry from a gmsh model as an array of shape
        (num_nodes, 3), where the i-th row corresponds to the i-th node in the
        mesh.

        Args:
            model: The GMSH model
            name: The name of the gmsh model. If not set the current model will be used

        Returns:
            The mesh geometry as an array of shape (num_nodes, 3).

        """
        if name is not None:
            model.setCurrent(name)

        # Get the unique tag and coordinates for nodes
        # in mesh
        indices, points, _ = model.mesh.getNodes()
        points = points.reshape(-1, 3)

        # GMSH indices starts at 1. We therefore subtract one to use zero-based numbering
        indices -= 1

        # In some cases, GMSH does not return the points in the same order as their unique node index.
        # We therefore sort nodes in geometry according to the unique index
        perm_sort = np.argsort(indices)
        assert np.all(indices[perm_sort] == np.arange(len(indices)))
        return points[perm_sort]

    # Map from Gmsh cell type identifier (integer) to DOLFINx cell type and degree
    # http://gmsh.info//doc/texinfo/gmsh.html#MSH-file-format
    _gmsh_to_cells = {1: ("interval", 1), 2: ("triangle", 1),
                      3: ("quadrilateral", 1), 4: ("tetrahedron", 1),
                      5: ("hexahedron", 1), 8: ("interval", 2),
                      9: ("triangle", 2), 10: ("quadrilateral", 2),
                      11: ("tetrahedron", 2), 12: ("hexahedron", 2),
                      15: ("point", 0), 21: ("triangle", 3),
                      26: ("interval", 3), 29: ("tetrahedron", 3),
                      36: ("quadrilateral", 3)}

    def ufl_mesh(gmsh_cell: int, gdim: int) -> ufl.Mesh:
        """Create a UFL mesh from a Gmsh cell identifier and the geometric dimension.
        See: http://gmsh.info//doc/texinfo/gmsh.html#MSH-file-format.

        Args:
            gmsh_cell: The Gmsh cell identifier
            gdim: The geometric dimension of the mesh

        Returns:
            A ufl Mesh using Lagrange elements (equispaced) of the corresponding DOLFINx cell
        """
        shape, degree = _gmsh_to_cells[gmsh_cell]
        cell = ufl.Cell(shape, geometric_dimension=gdim)
        scalar_element = ufl.FiniteElement("Lagrange", cell, degree, variant="equispaced")
        return ufl.Mesh(ufl.VectorElement(scalar_element))

    def cell_perm_array(cell_type: CellType, num_nodes: int) -> typing.List[int]:
        """The permuation array for permuting Gmsh ordering to DOLFINx ordering.

        Args:
            cell_type: The DOLFINx cell type
            num_nodes: The number of nodes in the cell

        Returns:
            An array `p` such that `a_dolfinx[i] = a_gmsh[p[i]]`.
        """
        return _cpp.io.perm_gmsh(cell_type, num_nodes)
