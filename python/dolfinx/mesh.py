# Copyright (C) 2017-2018 Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import types

import numpy

import ufl
from dolfinx import cpp


def locate_entities_geometrical(mesh: cpp.mesh.Mesh,
                                dim: int,
                                marker: types.FunctionType,
                                boundary_only: bool = False):
    """Compute list of mesh entities satisfying a geometric marking function.

    Parameters
    ----------
    mesh
        The mesh
    dim
        The topological dimension of the mesh entities to consider
    marker
        A function that takes an array of points `x` with shape
        ``(gdim, num_points)`` and returns an array of booleans of
        length ``num_points``, evaluating to `True` for entities whose
        degree-of-freedom should be returned.

    Returns
    -------
    numpy.ndarray
        Indices (local to the process) of marked mesh entities.

    """

    return cpp.mesh.locate_entities_geometrical(mesh, dim, marker, boundary_only)


_meshtags_types = {
    numpy.int8: cpp.mesh.MeshTags_int8,
    numpy.intc: cpp.mesh.MeshTags_int,
    numpy.int64: cpp.mesh.MeshTags_int64,
    numpy.double: cpp.mesh.MeshTags_double
}

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


def MeshTags(mesh, dim, indices, values, sorted=False, unique=False):

    if isinstance(values, int):
        values = numpy.full(indices.shape, values, dtype=numpy.intc)
    elif isinstance(values, float):
        values = numpy.full(indices.shape, values, dtype=numpy.double)

    dtype = values.dtype.type

    if dtype not in _meshtags_types.keys():
        raise KeyError("Datatype {} of values array not recognised".format(dtype))

    fn = _meshtags_types[dtype]
    return fn(mesh, dim, indices, values, sorted, unique)


def MeshFunction(value_type, mesh, dim, value):
    if value_type not in _meshfunction_types.keys():
        raise KeyError("MeshFunction type not recognised")
    fn = _meshfunction_types[value_type]
    return fn(mesh, dim, value)


def MeshValueCollection(value_type, mesh, dim=None, topology_data=None, value_data=None):
    if value_type not in _meshvaluecollection_types.keys():
        raise KeyError("MeshValueCollection type not recognised")
    mvc = _meshvaluecollection_types[value_type]

    if topology_data is not None:
        return mvc(mesh, dim, topology_data, value_data)

    if dim is not None:
        return mvc(mesh, dim)
    else:
        return mvc(mesh)


# Functions to extend cpp.mesh.Mesh with


def ufl_cell(self):
    return ufl.Cell(self.topology.cell_name(), geometric_dimension=self.geometry.dim)


def ufl_coordinate_element(self):
    """Return the finite element of the coordinate vector field of this
    domain.

    """
    # FIXME: This is all too implicit
    cell = self.ufl_cell()
    degree = self.geometry.dof_layout().degree()
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
