# Copyright (C) 2017-2018 Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import ufl
from dolfin import cpp

# __all__ = ["MeshFunction", "MeshValueCollection"]

_meshfunction_types = {
    "size_t": cpp.mesh.MeshFunctionSizet,
    "int": cpp.mesh.MeshFunctionInt,
    "double": cpp.mesh.MeshFunctionDouble
}

_meshvaluecollection_types = {
    "size_t": cpp.mesh.MeshValueCollection_sizet,
    "int": cpp.mesh.MeshValueCollection_int,
    "double": cpp.mesh.MeshValueCollection_double
}


class MeshFunction:
    def __new__(cls, value_type, mesh, dim, value):
        if value_type not in _meshfunction_types.keys():
            raise KeyError("MeshFunction type not recognised")
        fn = _meshfunction_types[value_type]
        return fn(mesh, dim, value)


class MeshValueCollection:
    def __new__(cls, value_type, mesh, dim=None):
        if value_type not in _meshvaluecollection_types.keys():
            raise KeyError("MeshValueCollection type not recognised")
        mvc = _meshvaluecollection_types[value_type]
        if dim is not None:
            return mvc(mesh, dim)
        else:
            return mvc(mesh)


# Functions to extend cpp.mesh.Mesh with


def ufl_cell(self):
    return ufl.Cell(self.cell_name(), geometric_dimension=self.geometry.dim)


def ufl_coordinate_element(self):
    """Return the finite element of the coordinate vector field of this
    domain.

    """
    cell = self.ufl_cell()
    degree = self.degree()
    return ufl.VectorElement(
        "Lagrange", cell, degree, dim=cell.geometric_dimension())


def ufl_domain(self):
    """Returns the ufl domain corresponding to the mesh."""
    # Cache object to avoid recreating it a lot
    if not hasattr(self, "_ufl_domain"):
        self._ufl_domain = ufl.Mesh(
            self.ufl_coordinate_element(), ufl_id=self.ufl_id(), cargo=self)
    return self._ufl_domain


def num_cells(self):
    """Return number of mesh cells"""
    return self.num_entities(self.topology.dim)


# Extend cpp.mesh.Mesh class, and clean-up
cpp.mesh.Mesh.ufl_cell = ufl_cell
cpp.mesh.Mesh.ufl_coordinate_element = ufl_coordinate_element
cpp.mesh.Mesh.ufl_domain = ufl_domain
cpp.mesh.Mesh.num_cells = num_cells

del ufl_cell, ufl_coordinate_element, ufl_domain, num_cells
