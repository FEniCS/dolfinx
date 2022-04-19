# Copyright (C) 2022 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tools to extract data from GMSH models"""
import numpy as np
import ufl
from dolfinx.cpp.io import perm_gmsh as cell_perm

__all__ = ["extract_topology_and_markers", "extract_geometry", "ufl_mesh", "cell_perm"]


def extract_topology_and_markers(gmsh_model, model_name=None):
    """Extract all entities tagged with a physical marker in the gmsh
    model, and collect the data per cell type. Returns a nested
    dictionary where the first key is the gmsh MSH element type integer.
    Each element type present in the model contains the cell topology of
    the elements and corresponding markers.

    """
    if model_name is not None:
        gmsh_model.setCurrent(model_name)

    # Get the physical groups from gmsh on the form [(dim1, tag1),(dim1,
    # tag2), (dim2, tag3),...]
    phys_grps = gmsh_model.getPhysicalGroups()
    topologies = {}
    for dim, tag in phys_grps:
        # Get the entities for a given dimension:
        # dim=0->Points, dim=1->Lines, dim=2->Triangles/Quadrilaterals,
        # etc.
        entities = gmsh_model.getEntitiesForPhysicalGroup(dim, tag)

        for entity in entities:
            # Get data about the elements on a given entity:
            # NOTE: Assumes that each entity only have one cell-type
            element_data = gmsh_model.mesh.getElements(dim, tag=entity)
            element_types, element_tags, node_tags = element_data
            assert len(element_types) == 1

            # The MSH type of the cells on the element
            element_type = element_types[0]
            num_el = len(element_tags[0])

            # Determine number of local nodes per element to create the
            # topology of the elements
            properties = gmsh_model.mesh.getElementProperties(element_type)
            name, dim, order, num_nodes, local_coords, _ = properties

            # 2D array of shape (num_elements,num_nodes_per_element)
            # containing the topology of the elements on this entity
            # NOTE: GMSH indexing starts with 1 and not zero
            element_topology = node_tags[0].reshape(-1, num_nodes) - 1

            # Gather data for each element type and the
            # corresponding physical markers
            if element_type in topologies.keys():
                topologies[element_type]["topology"] = np.concatenate(
                    (topologies[element_type]["topology"], element_topology), axis=0)
                topologies[element_type]["cell_data"] = np.concatenate(
                    (topologies[element_type]["cell_data"], np.full(num_el, tag)), axis=0)
            else:
                topologies[element_type] = {"topology": element_topology, "cell_data": np.full(num_el, tag)}

    return topologies


def extract_geometry(gmsh_model, model_name=None):
    """For a given gmsh model, extract the mesh geometry as a numpy
    (N, 3) array where the i-th row corresponds to the i-th node in the
    mesh.

    """
    if model_name is not None:
        gmsh_model.setCurrent(model_name)

    # Get the unique tag and coordinates for nodes
    # in mesh
    indices, points, _ = gmsh_model.mesh.getNodes()
    points = points.reshape(-1, 3)

    # GMSH indices starts at 1
    indices -= 1

    # Sort nodes in geometry according to the unique index
    perm_sort = np.argsort(indices)
    assert np.all(indices[perm_sort] == np.arange(len(indices)))
    return points[perm_sort]


# Map from Gmsh int to DOLFINx cell type and degree
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

    """
    shape, degree = _gmsh_to_cells[gmsh_cell]
    cell = ufl.Cell(shape, geometric_dimension=gdim)
    scalar_element = ufl.FiniteElement("Lagrange", cell, degree, variant="equispaced")
    return ufl.Mesh(ufl.VectorElement(scalar_element))
