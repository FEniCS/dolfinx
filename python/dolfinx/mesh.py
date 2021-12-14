# Copyright (C) 2017-2021 Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Creation, refining and marking of meshes"""

import types
import typing

import numpy as np

import ufl
from dolfinx import cpp as _cpp
from dolfinx.cpp.mesh import (CellType, GhostMode, build_dual_graph, cell_dim,
                              compute_boundary_facets,
                              compute_incident_entities, compute_midpoints,
                              create_cell_partitioner, create_meshtags)

from mpi4py import MPI as _MPI

__all__ = ["create_meshtags", "locate_entities", "locate_entities_boundary",
           "refine", "create_mesh", "create_meshtags", "MeshTags", "CellType",
           "GhostMode", "build_dual_graph", "cell_dim", "compute_midpoints",
           "compute_boundary_facets", "compute_incident_entities", "create_cell_partitioner"]


class Mesh(_cpp.mesh.Mesh):
    """A class for representing meshes. Mesh objects are not generally
    created using this class directly."""

    def __init__(self, comm: _MPI.Comm, topology: _cpp.mesh.Topology,
                 geometry: _cpp.mesh.Geometry, domain: ufl.Mesh):
        super().__init__(comm, topology, geometry)
        self._ufl_domain = domain
        domain._ufl_cargo = self

    @classmethod
    def from_cpp(cls, obj, domain: ufl.Mesh):
        """Create Mesh object from a C++ Mesh object"""
        obj._ufl_domain = domain
        obj.__class__ = Mesh
        domain._ufl_cargo = obj
        return obj

    def ufl_cell(self) -> ufl.Cell:
        """Return the UFL cell type"""
        return ufl.Cell(self.topology.cell_name(), geometric_dimension=self.geometry.dim)

    def ufl_domain(self) -> ufl.Mesh:
        """Return the ufl domain corresponding to the mesh."""
        return self._ufl_domain

    def sub(self, dim, entities):
        submesh = self.sub_without_ufl(dim, entities)
        # FIXME This is essentially a copy of the above ufl_cell method
        submesh_ufl_cell = ufl.Cell(submesh.topology.cell_name(),
                                    geometric_dimension=submesh.geometry.dim)
        # FIXME Don't hard code degree (and maybe Lagrange?)
        submesh_domain = ufl.Mesh(
            ufl.VectorElement("Lagrange",
                              cell=submesh_ufl_cell,
                              degree=1))
        return Mesh.from_cpp(submesh, submesh_domain)


def locate_entities(mesh: Mesh, dim: int, marker: types.FunctionType) -> np.ndarray:
    """Compute mesh entities satisfying a geometric marking function

    Parameters
    ----------
    mesh
        The mesh
    dim
        The topological dimension of the mesh entities to consider
    marker
        A function that takes an array of points `x` with shape ``(gdim,
        num_points)`` and returns an array of booleans of length
        ``num_points``, evaluating to `True` for entities to be located.

    Returns
    -------
    numpy.ndarray
        Indices (local to the process) of marked mesh entities.

    """

    return _cpp.mesh.locate_entities(mesh, dim, marker)


def locate_entities_boundary(mesh: Mesh, dim: int, marker: types.FunctionType) -> np.ndarray:
    """Compute mesh entities that are connected to an owned boundary
    facet and satisfy a geometric marking function

    For vertices and edges, in parallel this function will not
    necessarily mark all entities that are on the exterior boundary. For
    example, it is possible for a process to have a vertex that lies on
    the boundary without any of the attached facets being a boundary
    facet. When used to find degrees-of-freedom, e.g. using
    fem.locate_dofs_topological, the function that uses the data
    returned by this function must typically perform some parallel
    communication.

    Parameters
    ----------
    mesh
        The mesh
    dim
        The topological dimension of the mesh entities to consider
    marker
        A function that takes an array of points `x` with shape ``(gdim,
        num_points)`` and returns an array of booleans of length
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
    np.int8: _cpp.mesh.MeshTags_int8,
    np.int32: _cpp.mesh.MeshTags_int32,
    np.int64: _cpp.mesh.MeshTags_int64,
    np.double: _cpp.mesh.MeshTags_double
}


def refine(mesh: Mesh, edges: np.ndarray = None, redistribute: bool = True) -> Mesh:
    """Refine a mesh

    Parameters
    ----------
    mesh
        The mesh from which to build a refined mesh
    edges
        Optional argument to specify which edges should be refined. If
        not supplied uniform refinement is applied.
    redistribute
        Optional argument to redistribute the refined mesh if mesh is a
        distributed mesh.

    Returns
    -------
    Mesh
        A refined mesh
    """
    if edges is None:
        mesh_refined = _cpp.refinement.refine(mesh, redistribute)
    else:
        mesh_refined = _cpp.refinement.refine(mesh, edges, redistribute)

    coordinate_element = mesh._ufl_domain.ufl_coordinate_element()
    domain = ufl.Mesh(coordinate_element)
    return Mesh.from_cpp(mesh_refined, domain)


def create_mesh(comm: _MPI.Comm, cells: typing.Union[np.ndarray, _cpp.graph.AdjacencyList_int64],
                x: np.ndarray, domain: ufl.Mesh, ghost_mode=GhostMode.shared_facet,
                partitioner=_cpp.mesh.create_cell_partitioner()) -> Mesh:
    """
    Create a mesh from topology and geometry arrays

    comm
        The MPI communicator
    cells
        The cells of the mesh
    x
        The mesh geometry ('node' coordinates),  with shape ``(gdim,
        num_nodes)``
    domain
        The UFL mesh
    ghost_mode
        The ghost mode used in the mesh partitioning
    partitioner
        Function that computes the parallel distribution of cells across
        MPI ranks

    """
    ufl_element = domain.ufl_coordinate_element()
    cell_shape = ufl_element.cell().cellname()
    cell_degree = ufl_element.degree()
    cmap = _cpp.fem.CoordinateElement(_uflcell_to_dolfinxcell[cell_shape], cell_degree)
    try:
        mesh = _cpp.mesh.create_mesh(comm, cells, cmap, x, ghost_mode, partitioner)
    except TypeError:
        mesh = _cpp.mesh.create_mesh(comm, _cpp.graph.AdjacencyList_int64(np.cast['int64'](cells)),
                                     cmap, x, ghost_mode, partitioner)
    domain._ufl_cargo = mesh
    return Mesh.from_cpp(mesh, domain)


def MeshTags(mesh: Mesh, dim: int, indices: np.ndarray, values: np.ndarray) -> typing.Union[
        _cpp.mesh.MeshTags_double, _cpp.mesh.MeshTags_int32]:
    """Create a MeshTag for a set of mesh entities.

    Parameters
    ----------
    mesh
        The mesh
    dim
        The topological dimension of the mesh entity
    indices
        The entity indices (local to process)
    values
        The corresponding value for each entity
    """

    if isinstance(values, int):
        assert np.can_cast(values, np.int32)
        values = np.full(indices.shape, values, dtype=np.int32)
    elif isinstance(values, float):
        values = np.full(indices.shape, values, dtype=np.double)

    dtype = values.dtype.type
    if dtype not in _meshtags_types.keys():
        raise KeyError("Datatype {} of values array not recognised".format(dtype))

    fn = _meshtags_types[dtype]
    return fn(mesh, dim, indices.astype(np.int32), values)
