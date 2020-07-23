# Copyright (C) 2017-2020 Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import types
import typing

import numpy
import ufl
from dolfinx import cpp, fem


__all__ = [
    "locate_entities", "locate_entities_boundary", "refine", "create_mesh", "create_meshtags"
]


def locate_entities(mesh: cpp.mesh.Mesh,
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

    return cpp.mesh.locate_entities(mesh, dim, marker)


def locate_entities_boundary(mesh: cpp.mesh.Mesh,
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

    return cpp.mesh.locate_entities_boundary(mesh, dim, marker)


def refine(mesh, cell_markers=None, redistribute=True):
    """Refine a mesh"""
    if cell_markers is None:
        mesh_refined = cpp.refinement.refine(mesh, redistribute)
    else:
        mesh_refined = cpp.refinement.refine(mesh, cell_markers, redistribute)
    mesh_refined._ufl_domain = mesh._ufl_domain
    return mesh_refined


def create_mesh(comm, cells, x, domain, ghost_mode=cpp.mesh.GhostMode.shared_facet):
    """Create a mesh from topology and geometry data"""
    cmap = fem.create_coordinate_map(domain)
    try:
        mesh = cpp.mesh.create_mesh(comm, cells, cmap, x, ghost_mode)
    except TypeError:
        mesh = cpp.mesh.create_mesh(comm, cpp.graph.AdjacencyList_int64(numpy.cast['int64'](cells)),
                                    cmap, x, ghost_mode)

    # Attach UFL data (used when passing a mesh into UFL functions)
    domain._ufl_cargo = mesh
    mesh._ufl_domain = domain
    return mesh


_meshtags_types = {
    numpy.int8: cpp.mesh.MeshTags_int8,
    numpy.int32: cpp.mesh.MeshTags_int32,
    numpy.int64: cpp.mesh.MeshTags_int64,
    numpy.double: cpp.mesh.MeshTags_double
}


class MeshTags:
    def __init__(self, *args):
        """This initializer is not intended for the user interface. Use create_mesh_tags"""
        try:
            mesh, dim, indices, values = args
            if isinstance(values, int):
                values = numpy.full(indices.shape, values, dtype=numpy.intc)
            elif isinstance(values, float):
                values = numpy.full(indices.shape, values, dtype=numpy.double)
            dtype = values.dtype.type
            try:
                self._cpp_object = _meshtags_types[dtype](mesh, dim, indices, values)
            except KeyError:
                raise KeyError("Unsupported value type ({}) for MeshTags.".format(dtype))
        except ValueError:
            assert len(args) == 1
            self._cpp_object = args[0]

    @property
    def dim(self):
        return self._cpp_object.dim

    @property
    def mesh(self):
        return self._cpp_object.mesh

    @property
    def name(self):
        return self._cpp_object.name

    @name.setter
    def name(self, val):
        self._cpp_object.name = val

    @property
    def indices(self):
        return self._cpp_object.indices

    @property
    def values(self):
        return self._cpp_object.values

    def ufl_id(self):
        return self._cpp_object.id


def create_meshtags(mesh: cpp.mesh.Mesh, dim: int, indices: typing.Union[numpy.array, cpp.graph.AdjacencyList_int32],
                    values: numpy.array) -> MeshTags:
    """Create a MeshTags object
    Parameters
    ----------
    mesh
        The mesh
    dim
        The topological dimension of the mesh entities to tag
    indices
        The entities to tag. Can be an array of entity indices or an
        AdjacencyList with the vertices of each each tagged entity.

    Returns
    -------
    MeshTags
        MeshTags object.

    """
    try:
        # Create from an adjacency list of values
        return MeshTags(cpp.mesh.create_meshtags(mesh, dim, indices, values))
    except TypeError:
        # Create from an list of entity indices and values
        return MeshTags(mesh, dim, indices, values)


# Functions to extend cpp.mesh.Mesh with


def ufl_cell(self):
    return ufl.Cell(self.topology.cell_name(), geometric_dimension=self.geometry.dim)


def ufl_domain(self):
    """Return the ufl domain corresponding to the mesh."""
    return self._ufl_domain


# Extend cpp.mesh.Mesh class, and clean-up
cpp.mesh.Mesh.ufl_cell = ufl_cell
cpp.mesh.Mesh.ufl_domain = ufl_domain

del ufl_cell
del ufl_domain
