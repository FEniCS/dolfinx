# Copyright (C) 2017-2021 Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Creation, refining and marking of meshes"""

from __future__ import annotations

import typing

import numpy as np
import numpy.typing

import basix
import basix.ufl_wrapper
import ufl
from dolfinx import cpp as _cpp
from dolfinx.cpp.mesh import (CellType, DiagonalType, GhostMode,
                              build_dual_graph, cell_dim,
                              compute_incident_entities, compute_midpoints,
                              create_cell_partitioner, exterior_facet_indices,
                              to_string, to_type)

from mpi4py import MPI as _MPI

__all__ = ["meshtags_from_entities", "locate_entities", "locate_entities_boundary",
           "refine", "create_mesh", "Mesh", "MeshTagsMetaClass", "meshtags", "CellType",
           "GhostMode", "build_dual_graph", "cell_dim", "compute_midpoints",
           "exterior_facet_indices", "compute_incident_entities", "create_cell_partitioner",
           "create_interval", "create_unit_interval", "create_rectangle", "create_unit_square",
           "create_box", "create_unit_cube", "to_type", "to_string"]


class Mesh(_cpp.mesh.Mesh):
    def __init__(self, comm: _MPI.Comm, topology: _cpp.mesh.Topology,
                 geometry: _cpp.mesh.Geometry, domain: ufl.Mesh):
        """A class for representing meshes

        Args:
            comm: The MPI communicator
            topology: The mesh topology
            geometry: The mesh geometry
            domain: The MPI communicator

        Note:
            Mesh objects are not generally created using this class directly.

        """
        super().__init__(comm, topology, geometry)
        self._ufl_domain = domain
        domain._ufl_cargo = self

    @classmethod
    def from_cpp(cls, obj: _cpp.mesh.Mesh, domain: ufl.Mesh) -> Mesh:
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


def locate_entities(mesh: Mesh, dim: int, marker: typing.Callable) -> np.ndarray:
    """Compute mesh entities satisfying a geometric marking function

    Args:
        mesh: Mesh to locate entities on
        dim: Topological dimension of the mesh entities to consider
        marker: A function that takes an array of points `x` with shape ``(gdim,
            num_points)`` and returns an array of booleans of length
            ``num_points``, evaluating to `True` for entities to be located.

    Returns:
        Indices (local to the process) of marked mesh entities.

    """
    return _cpp.mesh.locate_entities(mesh, dim, marker)


def locate_entities_boundary(mesh: Mesh, dim: int, marker: typing.Callable) -> np.ndarray:
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

    Args:
        mesh: Mesh to locate boundary entities on
        dim: Topological dimension of the mesh entities to consider
        marker: Function that takes an array of points `x` with shape ``(gdim,
            num_points)`` and returns an array of booleans of length
            ``num_points``, evaluating to `True` for entities to be located.

    Returns:
        Indices (local to the process) of marked mesh entities.

    """
    return _cpp.mesh.locate_entities_boundary(mesh, dim, marker)


_uflcell_to_dolfinxcell = {
    "interval": CellType.interval,
    "interval2D": CellType.interval,
    "interval3D": CellType.interval,
    "triangle": CellType.triangle,
    "triangle3D": CellType.triangle,
    "quadrilateral": CellType.quadrilateral,
    "quadrilateral3D": CellType.quadrilateral,
    "tetrahedron": CellType.tetrahedron,
    "hexahedron": CellType.hexahedron
}


def refine(mesh: Mesh, edges: np.ndarray = None, redistribute: bool = True) -> Mesh:
    """Refine a mesh

    Args:
        mesh: The mesh from which to build a refined mesh
        edges: Optional argument to specify which edges should be refined. If
            not supplied uniform refinement is applied.
        redistribute:
            Optional argument to redistribute the refined mesh if mesh is a
            distributed mesh.

    Returns:
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
                x: np.ndarray, domain: ufl.Mesh,
                partitioner=_cpp.mesh.create_cell_partitioner(GhostMode.none)) -> Mesh:
    """
    Create a mesh from topology and geometry arrays

    Args:
        comm: MPI communicator to define the mesh on
        cells: Cells of the mesh
        x: Mesh geometry ('node' coordinates),  with shape ``(gdim, num_nodes)``
        domain: UFL mesh
        ghost_mode: The ghost mode used in the mesh partitioning
        partitioner: Function that computes the parallel distribution of cells across MPI ranks

    Returns:
        A new mesh

    """
    ufl_element = domain.ufl_coordinate_element()
    cell_shape = ufl_element.cell().cellname()
    cell_degree = ufl_element.degree()
    cmap = _cpp.fem.CoordinateElement(_uflcell_to_dolfinxcell[cell_shape], cell_degree)
    try:
        mesh = _cpp.mesh.create_mesh(comm, cells, cmap, x, partitioner)
    except TypeError:
        mesh = _cpp.mesh.create_mesh(comm, _cpp.graph.AdjacencyList_int64(np.cast['int64'](cells)),
                                     cmap, x, partitioner)
    domain._ufl_cargo = mesh
    return Mesh.from_cpp(mesh, domain)


def create_submesh(mesh, dim, entities):
    submesh, entity_map, vertex_map, geom_map = _cpp.mesh.create_submesh(mesh, dim, entities)
    submesh_ufl_cell = ufl.Cell(submesh.topology.cell_name(),
                                geometric_dimension=submesh.geometry.dim)
    submesh_domain = ufl.Mesh(basix.ufl_wrapper.create_vector_element(
        "Lagrange", submesh_ufl_cell.cellname(), mesh.geometry.cmap.degree, basix.LagrangeVariant.equispaced,
        dim=submesh.geometry.dim, gdim=submesh.geometry.dim))
    return (Mesh.from_cpp(submesh, submesh_domain), entity_map, vertex_map, geom_map)


# Add attribute to MeshTags
def _ufl_id(self) -> int:
    return id(self)


setattr(_cpp.mesh.MeshTags_int8, 'ufl_id', _ufl_id)
setattr(_cpp.mesh.MeshTags_int32, 'ufl_id', _ufl_id)
setattr(_cpp.mesh.MeshTags_int64, 'ufl_id', _ufl_id)
setattr(_cpp.mesh.MeshTags_float64, 'ufl_id', _ufl_id)

del _ufl_id


class MeshTagsMetaClass:
    def __init__(self, mesh: Mesh, dim: int, indices: numpy.typing.NDArray[typing.Any],
                 values: numpy.typing.NDArray[typing.Any]):
        """A distributed sparse matrix that uses compressed sparse row storage.

        Args:
            mesh: The mesh
            dim: Topological dimension of the mesh entity
            indices: Entity indices (local to process)
            values: The corresponding value for each entity

        Note:
            Objects of this type should be created using
            :func:`meshtags` and not created using this initialiser
            directly.

        """
        super().__init__(mesh, dim, indices.astype(np.int32), values)  # type: ignore

    def ufl_id(self) -> int:
        """Object identifier.

        Notes:
            This method is used by UFL.

        Returns:
            The `id` of the object

        """
        return id(self)


def meshtags(mesh: Mesh, dim: int, indices: np.ndarray,
             values: typing.Union[np.ndarray, int, float]) -> MeshTagsMetaClass:
    """Create a MeshTags object that associates data with a subset of mesh entities.

    Args:
        mesh: The mesh
        dim: Topological dimension of the mesh entity
        indices: Entity indices (local to process)
        values: The corresponding value for each entity

    Returns:
        A MeshTags object

    Note:
        The type of the returned MeshTags is inferred from the type of
        ``values``.

    """

    if isinstance(values, int):
        assert np.can_cast(values, np.int32)
        values = np.full(indices.shape, values, dtype=np.int32)
    elif isinstance(values, float):
        values = np.full(indices.shape, values, dtype=np.double)

    values = np.asarray(values)
    if values.dtype == np.int8:
        ftype = _cpp.mesh.MeshTags_int8
    elif values.dtype == np.int32:
        ftype = _cpp.mesh.MeshTags_int32
    elif values.dtype == np.int64:
        ftype = _cpp.mesh.MeshTags_int64
    elif values.dtype == np.float64:
        ftype = _cpp.mesh.MeshTags_float64
    else:
        raise NotImplementedError(f"Type {values.dtype} not supported.")

    tags = type("MeshTagsMetaClass", (MeshTagsMetaClass, ftype), {})
    return tags(mesh, dim, indices, values)


def meshtags_from_entities(mesh: Mesh, dim: int, entities: _cpp.graph.AdjacencyList_int32,
                           values: numpy.typing.NDArray[typing.Any]):
    """Create a MeshTags object that associates data with a subset of
    mesh entities, where the entities are defined by their vertices.

    Args:
        mesh: The mesh
        dim: Topological dimension of the mesh entity
        entities: Tagged entities, with entities defined by their vertices
        values: The corresponding value for each entity

    Returns:
        A MeshTags object

    Note:
        The type of the returned MeshTags is inferred from the type of
        ``values``.

    """

    if isinstance(values, int):
        assert np.can_cast(values, np.int32)
        values = np.full(entities.num_nodes, values, dtype=np.int32)
    elif isinstance(values, float):
        values = np.full(entities.num_nodes, values, dtype=np.double)

    values = np.asarray(values)
    return _cpp.mesh.create_meshtags(mesh, dim, entities, values)


def create_interval(comm: _MPI.Comm, nx: int, points: numpy.typing.ArrayLike, ghost_mode=GhostMode.shared_facet,
                    partitioner=None) -> Mesh:
    """Create an interval mesh

    Args:
        comm: MPI communicator
        nx: Number of cells
        points: Coordinates of the end points
        ghost_mode: Ghost mode used in the mesh partitioning. Options
            are `GhostMode.none' and `GhostMode.shared_facet`.
        partitioner: Partitioning function to use for determining the
            parallel distribution of cells across MPI ranks

    Returns:
        An interval mesh

    """
    if partitioner is None:
        partitioner = _cpp.mesh.create_cell_partitioner(ghost_mode)
    domain = ufl.Mesh(basix.ufl_wrapper.create_vector_element(
        "Lagrange", "interval", 1, basix.LagrangeVariant.equispaced))
    mesh = _cpp.mesh.create_interval(comm, nx, points, ghost_mode, partitioner)
    return Mesh.from_cpp(mesh, domain)


def create_unit_interval(comm: _MPI.Comm, nx: int, ghost_mode=GhostMode.shared_facet,
                         partitioner=None) -> Mesh:
    """Create a mesh on the unit interval

    Args:
        comm: MPI communicator
        nx: Number of cells
        points: Coordinates of the end points
        ghost_mode: Ghost mode used in the mesh partitioning. Options
            are `GhostMode.none' and `GhostMode.shared_facet`.
        partitioner: Partitioning function to use for determining the
            parallel distribution of cells across MPI ranks

    Returns:
        A unit interval mesh with end points at 0 and 1

    """
    if partitioner is None:
        partitioner = _cpp.mesh.create_cell_partitioner(ghost_mode)
    return create_interval(comm, nx, [0.0, 1.0], ghost_mode, partitioner)


def create_rectangle(comm: _MPI.Comm, points: numpy.typing.ArrayLike, n: numpy.typing.ArrayLike,
                     cell_type=CellType.triangle, ghost_mode=GhostMode.shared_facet,
                     partitioner=None,
                     diagonal: DiagonalType = DiagonalType.right) -> Mesh:
    """Create rectangle mesh

    Args:
        comm: MPI communicator
        points: Coordinates of the lower-left and upper-right corners of the
            rectangle
        n: Number of cells in each direction
        cell_type: Mesh cell type
        ghost_mode: Ghost mode used in the mesh partitioning
        partitioner: Function that computes the parallel distribution of
            cells across MPI ranks
        diagonal: Direction of diagonal of triangular meshes. The
            options are ``left``, ``right``, ``crossed``, ``left/right``,
            ``right/left``.

    Returns:
        A mesh of a rectangle

    """
    if partitioner is None:
        partitioner = _cpp.mesh.create_cell_partitioner(ghost_mode)
    domain = ufl.Mesh(basix.ufl_wrapper.create_vector_element(
        "Lagrange", cell_type.name, 1, basix.LagrangeVariant.equispaced))
    mesh = _cpp.mesh.create_rectangle(comm, points, n, cell_type, partitioner, diagonal)

    return Mesh.from_cpp(mesh, domain)


def create_unit_square(comm: _MPI.Comm, nx: int, ny: int, cell_type=CellType.triangle,
                       ghost_mode=GhostMode.shared_facet, partitioner=None,
                       diagonal: DiagonalType = DiagonalType.right) -> Mesh:
    """Create a mesh of a unit square

    Args:
        comm: MPI communicator
        nx: Number of cells in the "x" direction
        ny: Number of cells in the "y" direction
        cell_type: Mesh cell type
        ghost_mode: Ghost mode used in the mesh partitioning
        partitioner:Function that computes the parallel distribution of cells across
            MPI ranks
        diagonal:
            Direction of diagonal

    Returns:
        A mesh of a square with corners at (0, 0) and (1, 1)

    """
    if partitioner is None:
        partitioner = _cpp.mesh.create_cell_partitioner(ghost_mode)
    return create_rectangle(comm, [np.array([0.0, 0.0]),
                                   np.array([1.0, 1.0])], [nx, ny], cell_type, ghost_mode,
                            partitioner, diagonal)


def create_box(comm: _MPI.Comm, points: typing.List[numpy.typing.ArrayLike], n: list,
               cell_type=CellType.tetrahedron,
               ghost_mode=GhostMode.shared_facet,
               partitioner=None) -> Mesh:
    """Create box mesh

    Args:
        comm: MPI communicator
        points: Coordinates of the 'lower-left' and 'upper-right'
            corners of the box
        n: List of cells in each direction
        cell_type: The cell type
        ghost_mode: The ghost mode used in the mesh partitioning
        partitioner: Function that computes the parallel distribution of
            cells across MPI ranks

    Returns:
        A mesh of a box domain

    """
    if partitioner is None:
        partitioner = _cpp.mesh.create_cell_partitioner(ghost_mode)
    domain = ufl.Mesh(basix.ufl_wrapper.create_vector_element(
        "Lagrange", cell_type.name, 1, basix.LagrangeVariant.equispaced))
    mesh = _cpp.mesh.create_box(comm, points, n, cell_type, partitioner)

    return Mesh.from_cpp(mesh, domain)


def create_unit_cube(comm: _MPI.Comm, nx: int, ny: int, nz: int, cell_type=CellType.tetrahedron,
                     ghost_mode=GhostMode.shared_facet, partitioner=None) -> Mesh:
    """Create a mesh of a unit cube

    Args:
        comm: MPI communicator
        nx: Number of cells in "x" direction
        ny: Number of cells in "y" direction
        nz: Number of cells in "z" direction
        cell_type: Mesh cell type
        ghost_mode: Ghost mode used in the mesh partitioning
        partitioner: Function that computes the parallel distribution of
            cells across MPI ranks

    Returns:
        A mesh of an axis-aligned unit cube with corners at (0, 0, 0)
        and (1, 1, 1)

    """
    if partitioner is None:
        partitioner = _cpp.mesh.create_cell_partitioner(ghost_mode)
    return create_box(comm, [np.array([0.0, 0.0, 0.0]), np.array(
        [1.0, 1.0, 1.0])], [nx, ny, nz], cell_type, ghost_mode, partitioner)
