# Copyright (C) 2017-2020 Chris N. Richardson, Garth N. Wells, Michal Habera
# and Jørgen S. Dokken
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""IO module for input data, post-processing and checkpointing"""

import ufl
from dolfinx import cpp, fem
from dolfinx.mesh import MeshTags


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

    def write_meshtags(self, tags, **kwargs):
        tags_cpp = tags._cpp_object
        super().write_meshtags(tags_cpp, **kwargs)

    def read_meshtags(self, mesh, tagname, **kwargs):
        tags_cpp = super().read_meshtags(mesh, tagname, **kwargs)
        mt = MeshTags(tags_cpp.mesh, tags_cpp.dim, tags_cpp.indices, tags_cpp.values)
        mt._cpp_object = tags_cpp
        mt.indices = mt._cpp_object.indices
        mt.values = mt._cpp_object.values
        return mt

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


# Map from Gmsh string to DOLFIN cell type and degree
_gmsh_cells = dict(tetra=("tetrahedron", 1), tetra10=("tetrahedron", 2), tetra20=("tetrahedron", 3),
                   hexahedron=("hexahedron", 1), hexahedron27=("hexahedron", 2),
                   triangle=("triangle", 1), triangle6=("triangle", 2), triangle10=("triangle", 3),
                   quad=("quadrilateral", 1), quad9=("quadrilateral", 2), quad16=("quadrilateral", 3),
                   line=("interval", 1), line3=("interval", 2), line4=("interval", 3),
                   vertex=("point", 1))


def ufl_mesh_from_gmsh(gmsh_cell, gdim):
    """Create a UFL mesh from a Gmsh cell string and the geometric dimension."""
    shape, degree = _gmsh_cells[gmsh_cell]
    cell = ufl.Cell(shape, geometric_dimension=gdim)
    return ufl.Mesh(ufl.VectorElement("Lagrange", cell, degree))
