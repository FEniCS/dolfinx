# Copyright (C) 2017-2018 Chris N. Richardson, Garth N. Wells and Michal Habera
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""IO module for input data, post-processing and checkpointing"""

import ufl
from dolfinx import cpp, fem


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

    def read_mesh(self, name="mesh", xpath="/Xdmf/Domain"):
        # Read mesh data from file
        cell_type, x, cells = super().read_mesh_data(name, xpath)

        # Construct the geometry map
        cell = ufl.Cell(cpp.mesh.to_string(cell_type[0]), geometric_dimension=x.shape[1])
        domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, cell_type[1]))
        cmap = fem.create_coordinate_map(domain)

        # Build the mesh
        mesh = cpp.mesh.create(self.comm(), cpp.graph.AdjacencyList64(cells), cmap, x, cpp.mesh.GhostMode.none)
        mesh.name = name
        domain._ufl_cargo = mesh
        mesh._ufl_domain = domain

        return mesh
