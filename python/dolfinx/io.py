# Copyright (C) 2017-2020 Chris N. Richardson, Garth N. Wells, Michal Habera
# and Jørgen S. Dokken
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""IO module for input data, post-processing and checkpointing"""

import ufl
import numpy
from dolfinx import cpp, fem
from mpi4py import MPI
import dolfinx.mesh


class VTKFile:
    """Interface to VTK files
    VTK supports arbitrary order Lagrangian finite elements for the
    geometry description. XDMF is the preferred format for geometry
    order <= 2.

    """

    def __init__(self, filename: str):
        """Open VTK file
        Parameters
        ----------
        filename
            Name of the file
        """
        self._cpp_object = cpp.io.VTKFile(filename)

    def write(self, o, t=None) -> None:
        """Write object to file"""
        o_cpp = getattr(o, "_cpp_object", o)
        if t is None:
            self._cpp_object.write(o_cpp)
        else:
            self._cpp_object.write(o_cpp, t)


class XDMFFile(cpp.io.XDMFFile):
    def write_function(self, u, t=0.0, mesh_xpath="/Xdmf/Domain/Grid[@GridType='Uniform'][1]"):
        u_cpp = getattr(u, "_cpp_object", u)
        super().write_function(u_cpp, t, mesh_xpath)

    def read_mesh(self, ghost_mode=cpp.mesh.GhostMode.shared_facet, name="mesh", xpath="/Xdmf/Domain"):
        # Read mesh data from file
        cell_type = super().read_cell_type(name, xpath)
        cells = super().read_topology_data(name, xpath)
        x = super().read_geometry_data(name, xpath)

        # Construct the geometry map
        cell = ufl.Cell(cpp.mesh.to_string(cell_type[0]), geometric_dimension=x.shape[1])
        domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, cell_type[1]))
        cmap = fem.create_coordinate_map(domain)

        # Build the mesh
        mesh = cpp.mesh.create_mesh(self.comm(), cpp.graph.AdjacencyList_int64(cells), cmap, x, ghost_mode)
        mesh.name = name
        domain._ufl_cargo = mesh
        mesh._ufl_domain = domain

        return mesh


def extract_mesh_topology_and_markers(gmsh_model, model_name=None):
    """
    Extracts all entities tagged with a physical marker
    in the gmsh model, and collects the data per cell type.
    Returns a nested dictionary where the first key is the gmsh
    MSH element type integer. Each element type present
    in the model contains the cell topology of the elements
    and corresponding markers.
    Input:
          gmsh_model - The GMSH model
          model_name - Name of model (Default: None)
    Example:
    MSH_triangle=2
    MSH_tetra=4
    topologies = {MSH_triangle: {"topology": triangle_topology,
                             "cell_data": triangle_markers},
              MSH_tetra: {"topology": tetra_topology,
                          "cell_data": tetra_markers}}
    """
    if model_name is not None:
        gmsh_model.setCurrent(model_name)
    # Get the physical groups from gmsh on the form
    # [(dim1, tag1),(dim1, tag2), (dim2, tag3),...]
    phys_grps = gmsh_model.getPhysicalGroups()
    topologies = {}
    for dim, tag in phys_grps:
        # Get the entities for a given dimension:
        # dim=0->Points, dim=1->Lines, dim=2->Triangles/Quadrilaterals
        # etc.
        entities = gmsh_model.getEntitiesForPhysicalGroup(dim, tag)

        for entity in entities:
            # Get data about the elements on a given entity:
            # NOTE: Assumes that each entity only have one cell-type
            element_data = gmsh_model.mesh.getElements(dim, tag=entity)
            element_types, element_tags, node_tags = element_data
            assert(len(element_types) == 1)
            # The MSH type of the cells on the element
            element_type = element_types[0]
            num_el = len(element_tags[0])
            # Determine number of local nodes per element to create
            # the topology of the elements
            properties = gmsh_model.mesh.getElementProperties(element_type)
            name, dim, order, num_nodes, local_coords, _ = properties
            # 2D array of shape (num_elements,num_nodes_per_element) containing
            # the topology of the elements on this entity
            # NOTE: GMSH indexing starts with 1 and not zero.
            element_topology = node_tags[0].reshape(-1, num_nodes) - 1

            # Gather data for each element type and the
            # corresponding physical markers
            if element_type in topologies.keys():
                topologies[element_type]["topology"] =\
                    numpy.concatenate((topologies[element_type]["topology"],
                                       element_topology), axis=0)
                topologies[element_type]["cell_data"] =\
                    numpy.concatenate((topologies[element_type]["cell_data"],
                                       numpy.full(num_el, tag)), axis=0)
            else:
                topologies[element_type] = {"topology": element_topology,
                                            "cell_data": numpy.full(num_el, tag)}

    return topologies


def extract_geometry(gmsh_model, model_name=None):
    """
    For a given gmsh model, extract the mesh geometry
    as a numpy (N,3) array where the i-th row
    corresponds to the i-th node in the mesh
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
    perm_sort = numpy.argsort(indices)
    assert numpy.all(indices[perm_sort] == numpy.arange(len(indices)))
    return points[perm_sort]


def read_from_msh(filename, cell_data=False, facet_data=False, gdim=None):
    """
    Reads a mesh from a msh-file and returns the dolfin-x mesh.
    Input:
        filename: Name of msh file
        cell_data: Boolean, True of a mesh tag for cell data should be returned
                   (Default: False)
        facet_data: Boolean, True if a mesh tag for facet data should be
                    returned (Default: False)
        gdim: Geometrical dimension of problem (Default: 3)
    """
    if gdim is None:
        gdim = 3
    if MPI.COMM_WORLD.rank == 0:
        import gmsh
        gmsh.initialize()

        gmsh.merge(filename)

        # Get mesh geometry
        x = extract_geometry(gmsh.model)

        # Get mesh topology for each element
        topologies = extract_mesh_topology_and_markers(gmsh.model)

        # Get information about each cell type from the msh files
        num_cell_types = len(topologies.keys())
        cell_information = {}
        cell_dimensions = numpy.zeros(num_cell_types, dtype=numpy.int32)
        for i, element in enumerate(topologies.keys()):
            properties = gmsh.model.mesh.getElementProperties(element)
            name, dim, order, num_nodes, local_coords, _ = properties
            cell_information[i] = {"id": element, "dim": dim,
                                   "num_nodes": num_nodes}
            cell_dimensions[i] = dim
        gmsh.finalize()

        # Sort elements by ascending dimension
        perm_sort = numpy.argsort(cell_dimensions)

        # Broadcast cell type data and geometric dimension
        cell_id = cell_information[perm_sort[-1]]["id"]
        tdim = cell_information[perm_sort[-1]]["dim"]
        num_nodes = cell_information[perm_sort[-1]]["num_nodes"]
        cell_id, num_nodes = MPI.COMM_WORLD.bcast([cell_id, num_nodes], root=0)

        # Check for facet data and broadcast if found
        if facet_data:
            if tdim - 1 in cell_dimensions:
                num_facet_nodes = MPI.COMM_WORLD.bcast(
                    cell_information[perm_sort[-2]]["num_nodes"], root=0)
                gmsh_facet_id = cell_information[perm_sort[-2]]["id"]
                marked_facets = topologies[gmsh_facet_id]["topology"]
                facet_values = topologies[gmsh_facet_id]["cell_data"]
            else:
                raise ValueError("No facet data found in file.")

        cells = topologies[cell_id]["topology"]
        cell_values = topologies[cell_id]["cell_data"]

    else:
        cell_id, num_nodes = MPI.COMM_WORLD.bcast([None, None], root=0)
        cells, x = numpy.empty([0, num_nodes]), numpy.empty([0, gdim])
        cell_values = numpy.empty((0,))
        if facet_data:
            num_facet_nodes = MPI.COMM_WORLD.bcast(None, root=0)
            marked_facets = numpy.empty((0, num_facet_nodes))
            facet_values = numpy.empty((0,))

    # Create distributed mesh
    mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, x[:, :gdim],
                                    ufl_mesh_from_gmsh(cell_id, gdim))
    # Create MeshTags for cells
    if cell_data:
        local_entities, local_values = cpp.io.extract_local_entities(
            mesh, mesh.topology.dim, cells, cell_values)
        mesh.topology.create_connectivity(mesh.topology.dim, 0)
        adj = cpp.graph.AdjacencyList_int32(local_entities)
        ct = dolfinx.mesh.create_meshtags(mesh, mesh.topology.dim,
                                          adj, numpy.int32(local_values))
        ct.name = "Cell tags"
    # Create MeshTags for facets
    if facet_data:
        local_entities, local_values = cpp.io.extract_local_entities(
            mesh, mesh.topology.dim - 1, marked_facets, facet_values)
        mesh.topology.create_connectivity(
            mesh.topology.dim - 1, mesh.topology.dim)
        adj = cpp.graph.AdjacencyList_int32(local_entities)
        ft = dolfinx.mesh.create_meshtags(mesh, mesh.topology.dim - 1,
                                          adj, numpy.int32(local_values))
        ft.name = "Facet tags"

    if cell_data and facet_data:
        return mesh, ct, ft
    elif cell_data and not facet_data:
        return mesh, ct
    elif not cell_data and facet_data:
        return mesh, ft
    else:
        return mesh


# Map from Gmsh float to DOLFIN cell type and degree
# http://gmsh.info//doc/texinfo/gmsh.html#MSH-file-format
_gmsh_to_cells = {1: ("interval", 1), 2: ("triangle", 1),
                  3: ("quadrilateral", 1), 4: ("tetrahedron", 1),
                  5: ("hexahedron", 1), 8: ("interval", 2),
                  9: ("triangle", 2), 10: ("quadrilateral", 2),
                  11: ("tetrahedron", 2), 12: ("hexahedron", 2),
                  15: ("point", 0), 21: ("triangle", 3),
                  26: ("interval", 3), 29: ("tetrahedron", 3),
                  36: ("quadrilateral", 3)}


def ufl_mesh_from_gmsh(gmsh_cell, gdim):
    """Create a UFL mesh from a Gmsh cell int and the geometric dimension."""
    shape, degree = _gmsh_to_cells[gmsh_cell]
    cell = ufl.Cell(shape, geometric_dimension=gdim)
    return ufl.Mesh(ufl.VectorElement("Lagrange", cell, degree))
