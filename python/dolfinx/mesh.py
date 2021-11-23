# Copyright (C) 2017-2020 Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Creation, refining and marking of meshes"""

import types

import numpy
import ufl

from dolfinx import cpp as _cpp
from dolfinx.cpp.mesh import CellType  # noqa
from dolfinx.cpp.mesh import GhostMode  # noqa
from dolfinx.cpp.mesh import midpoints  # noqa
from dolfinx.cpp.mesh import create_meshtags

__all__ = [
    "locate_entities", "locate_entities_boundary", "refine", "create_mesh", "create_meshtags", "MeshTags"
]


def locate_entities(mesh: _cpp.mesh.Mesh,
                    dim: int,
                    marker: types.FunctionType):
    """Compute list of mesh entities satisfying a geometric marking function.

    Parameters
    ----------
    mesh
        The mesh
    dim
        The topological dimension of the mesh entities to consider
    marker
        A function that takes an array of points `x` with shape
        ``(gdim, num_points)`` and returns an array of booleans of length
        ``num_points``, evaluating to `True` for entities to be located.

    Returns
    -------
    numpy.ndarray
        Indices (local to the process) of marked mesh entities.

    """

    return _cpp.mesh.locate_entities(mesh, dim, marker)


def locate_entities_boundary(mesh: _cpp.mesh.Mesh,
                             dim: int,
                             marker: types.FunctionType):
    """Compute list of mesh entities that are attached to an owned boundary facet
    and satisfy a geometric marking function.

    For vertices and edges, in parallel this function will not necessarily
    mark all entities that are on the exterior boundary. For example, it is
    possible for a process to have a vertex that lies on the boundary without
    any of the attached facets being a boundary facet. When used to find
    degrees-of-freedom, e.g. using fem.locate_dofs_topological, the function
    that uses the data returned by this function must typically perform some
    parallel communication.

    Parameters
    ----------
    mesh
        The mesh
    dim
        The topological dimension of the mesh entities to
        consider
    marker
        A function that takes an array of points `x` with shape
        ``(gdim, num_points)`` and returns an array of booleans of length
        ``num_points``, evaluating to `True` for entities to be located.

    Returns
    -------
    numpy.ndarray
        Indices (local to the process) of marked mesh entities.

    """

    return _cpp.mesh.locate_entities_boundary(mesh, dim, marker)


_uflcell_to_dolfinxcell = {
    "interval": CellType.interval,
    "triangle": CellType.triangle,
    "quadrilateral": CellType.quadrilateral,
    "tetrahedron": CellType.tetrahedron,
    "hexahedron": CellType.hexahedron
}

_meshtags_types = {
    numpy.int8: _cpp.mesh.MeshTags_int8,
    numpy.int32: _cpp.mesh.MeshTags_int32,
    numpy.int64: _cpp.mesh.MeshTags_int64,
    numpy.double: _cpp.mesh.MeshTags_double
}


def refine(mesh, cell_markers=None, redistribute=True):
    """Refine a mesh"""
    if cell_markers is None:
        mesh_refined = _cpp.refinement.refine(mesh, redistribute)
    else:
        mesh_refined = _cpp.refinement.refine(mesh, cell_markers, redistribute)

    coordinate_element = mesh._ufl_domain.ufl_coordinate_element()
    domain = ufl.Mesh(coordinate_element)
    domain._ufl_cargo = mesh_refined
    mesh_refined._ufl_domain = domain
    return mesh_refined


def create_mesh(comm, cells, x, domain,
                ghost_mode=GhostMode.shared_facet,
                partitioner=_cpp.mesh.partition_cells_graph):
    """Create a mesh from topology and geometry data"""
    ufl_element = domain.ufl_coordinate_element()
    cell_shape = ufl_element.cell().cellname()
    cell_degree = ufl_element.degree()
    cmap = _cpp.fem.CoordinateElement(_uflcell_to_dolfinxcell[cell_shape], cell_degree)
    try:
        mesh = _cpp.mesh.create_mesh(comm, cells, cmap, x, ghost_mode, partitioner)
    except TypeError:
        mesh = _cpp.mesh.create_mesh(comm, _cpp.graph.AdjacencyList_int64(numpy.cast['int64'](cells)),
                                     cmap, x, ghost_mode, partitioner)

    # Attach UFL data (used when passing a mesh into UFL functions)
    domain._ufl_cargo = mesh
    mesh._ufl_domain = domain
    return mesh


def MeshTags(mesh, dim, indices, values):

    if isinstance(values, int):
        values = numpy.full(indices.shape, values, dtype=numpy.int32)
    elif isinstance(values, float):
        values = numpy.full(indices.shape, values, dtype=numpy.double)

    dtype = values.dtype.type
    if dtype not in _meshtags_types.keys():
        raise KeyError("Datatype {} of values array not recognised".format(dtype))

    fn = _meshtags_types[dtype]
    return fn(mesh, dim, indices.astype(numpy.int32), values)


class Mesh:
    def __init__(self, mesh: _cpp.mesh.Mesh):
        self._cpp_object = mesh

    @property
    def topology(self):
        return self._cpp_object.topology

    @property
    def geometry(self):
        return self._cpp_object.geometry

    @property
    def comm(self):
        return self._cpp_object.mpi_comm

    def ufl_cell(self):
        return ufl.Cell(self.topology.cell_name(), geometric_dimension=self.geometry.dim)

    def ufl_domain(self):
        """Return the ufl domain corresponding to the mesh."""
        return self._ufl_domain


# Functions to extend cpp.mesh.Mesh with


def ufl_cell(self):
    return ufl.Cell(self.topology.cell_name(), geometric_dimension=self.geometry.dim)


def ufl_domain(self):
    """Return the ufl domain corresponding to the mesh."""
    return self._ufl_domain


# Extend cpp.mesh.Mesh class, and clean-up
_cpp.mesh.Mesh.ufl_cell = ufl_cell
_cpp.mesh.Mesh.ufl_domain = ufl_domain

del ufl_cell
del ufl_domain
