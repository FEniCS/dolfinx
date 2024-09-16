# Copyright (C) 2017-2021 Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Creation, refining and marking of meshes"""

from __future__ import annotations

import typing

from mpi4py import MPI as _MPI

import numpy as np
import numpy.typing as npt

import basix
import basix.ufl
import ufl
from dolfinx import cpp as _cpp
from dolfinx import default_real_type
from dolfinx.cpp.mesh import (
    CellType,
    DiagonalType,
    GhostMode,
    build_dual_graph,
    cell_dim,
    create_cell_partitioner,
    to_string,
    to_type,
)
from dolfinx.cpp.refinement import RefinementOption
from dolfinx.fem import CoordinateElement as _CoordinateElement
from dolfinx.fem import coordinate_element as _coordinate_element

__all__ = [
    "meshtags_from_entities",
    "locate_entities",
    "locate_entities_boundary",
    "refine",
    "create_mesh",
    "create_submesh",
    "Mesh",
    "MeshTags",
    "meshtags",
    "CellType",
    "GhostMode",
    "build_dual_graph",
    "cell_dim",
    "compute_midpoints",
    "exterior_facet_indices",
    "compute_incident_entities",
    "create_cell_partitioner",
    "create_interval",
    "create_unit_interval",
    "create_rectangle",
    "create_unit_square",
    "create_box",
    "create_unit_cube",
    "to_type",
    "to_string",
    "refine_interval",
    "refine_plaza",
    "transfer_meshtag",
    "entities_to_geometry",
]


class Topology:
    """Topology for a :class:`dolfinx.mesh.Mesh`"""

    _cpp_object: _cpp.mesh.Topology

    def __init__(self, topology: _cpp.mesh.Topology):
        """Initialize a topology from a C++ topology.
        Args: The C++ topology object
        Note:
            topology objects should not usually be created using this
            initializer directly.
        """
        self._cpp_object = topology

    def cell_name(self) -> str:
        """String representation of the cell-type of the topology"""
        return to_string(self._cpp_object.cell_type)

    def connectivity(self, d0: int, d1: int) -> _cpp.graph.AdjacencyList_int32:
        """Return connectivity from entities of dimension ``d0`` to entities of dimension ``d1``.

        Note:
            Assumes only one entity type per dimension.

        Args:
            d0: Dimension of entity one is mapping from
            d1: Dimension of entity one is mapping to
        """
        if (conn := self._cpp_object.connectivity(d0, d1)) is not None:
            return conn
        else:
            raise RuntimeError(
                f"Connectivity between dimension {d0} and {d1} has not been computed.",
                "Please call :func:`dolfinx.mesh.Topology.create_connectivity` first.",
            )

    @property
    def comm(self):
        return self._cpp_object.comm

    def create_connectivity(self, d0: int, d1: int):
        """Create connectivity between given pair of dimensions, ``d0`` and  ``d1``.

        Args:
            d0: Dimension of entities one is mapping from
            d1: Dimension of entities one is mapping to
        """
        self._cpp_object.create_connectivity(d0, d1)

    def create_entities(self, dim: int) -> int:
        """Create entities of given topological dimension.

        Args:
            dim: Topological dimension

        Returns:
            Number of newly created entities, returns -1 if entities already existed
        """
        return self._cpp_object.create_entities(dim)

    def create_entity_permutations(self):
        """Compute entity permutations and reflections."""
        self._cpp_object.create_entity_permutations()

    @property
    def dim(self) -> int:
        """Return the topological dimension of the mesh."""
        return self._cpp_object.dim

    @property
    def entity_types(self) -> list[list[CellType]]:
        """Get the entity types in the topology for all topological dimensions."""
        return self._cpp_object.entity_types

    def get_cell_permutation_info(self) -> npt.NDArray[np.uint32]:
        """Returns the permutation information"""
        return self._cpp_object.get_cell_permutation_info()

    def get_facet_permutations(self) -> npt.NDArray[np.uint8]:
        """Get the permutation number to apply to facets."""
        return self._cpp_object.get_facet_permutations()

    def index_map(self, dim: int) -> _cpp.common.IndexMap:
        """Get the IndexMap that described the parallel distribution of the mesh entities.

        Args:
            dim: Topological dimension

        Returns:
            Index map for the entities of dimension `dim`.
        """
        if (imap := self._cpp_object.index_map(dim)) is not None:
            return imap
        else:
            raise RuntimeError(f"Entities of dimension {dim} has not been computed")

    def interprocess_facets(self) -> npt.NDArray[np.int32]:
        """List of inter-process facets, if facet topology has been computed."""
        return self._cpp_object.interprocess_facets()

    @property
    def original_cell_index(self) -> npt.NDArray[np.int64]:
        """Get the original cell index"""
        return self._cpp_object.original_cell_index

    def set_connectivity(self, graph: _cpp.graph.AdjacencyList_int32, d0: int, d1: int):
        """Set connectivity for given pair of topological dimensions.

        Args:
            graph: Connectivity graph
            d0: Topological dimension mapping from
            d1: Topological dimension mapping to
        """
        self._cpp_object.set_connectivity(graph, d0, d1)

    def set_index_map(self, dim: int, index_map: _cpp.common.IndexMap):
        """Set the IndexMap for dimension ``dim``.

        Args:
            dim: Topological dimension of entity
            index_map: IndexMap
        """
        return self._cpp_object.set_index_map(dim, index_map)

    @property
    def cell_type(self) -> CellType:
        """Get the cell type of the topology"""
        return self._cpp_object.cell_type


class Mesh:
    """A mesh."""

    _mesh: typing.Union[_cpp.mesh.Mesh_float32, _cpp.mesh.Mesh_float64]
    _topology: Topology
    _ufl_domain: typing.Optional[ufl.Mesh]

    def __init__(self, mesh, domain: typing.Optional[ufl.Mesh]):
        """Initialize mesh from a C++ mesh.

        Args:
            mesh: A C++ mesh object.
            domain: A UFL domain.

        Note:
            Mesh objects should not usually be created using this
            initializer directly.
        """
        self._cpp_object = mesh
        self._topology = Topology(self._cpp_object.topology)
        self._ufl_domain = domain
        if self._ufl_domain is not None:
            self._ufl_domain._ufl_cargo = self._cpp_object  # type: ignore

    @property
    def comm(self):
        return self._cpp_object.comm

    @property
    def name(self):
        return self._cpp_object.name

    @name.setter
    def name(self, value):
        self._cpp_object.name = value

    def ufl_cell(self) -> ufl.Cell:
        """Return the UFL cell type.

        Note:
            This method is required for UFL compatibility.
        """
        return ufl.Cell(self.topology.cell_name())

    def ufl_domain(self) -> ufl.Mesh:
        """Return the ufl domain corresponding to the mesh.

        Domain is ``None`` if it has not been set.

        Note:
            This method is required for UFL compatibility.
        """
        return self._ufl_domain

    def basix_cell(self) -> ufl.Cell:
        """Return the Basix cell type."""
        return getattr(basix.CellType, self.topology.cell_name())

    def h(self, dim: int, entities: npt.NDArray[np.int32]) -> npt.NDArray[np.float64]:
        """Geometric size measure of cell entities.

        Args:
            dim: Topological dimension of the entities to compute the
                size measure of.
            entities: Indices of entities of dimension ``dim`` to
                compute size measure of.

        Returns:
            Size measure for each requested entity.
        """
        return _cpp.mesh.h(self._cpp_object, dim, entities)

    @property
    def topology(self) -> Topology:
        "Mesh topology."
        return self._topology

    @property
    def geometry(self):
        "Mesh geometry."
        return self._cpp_object.geometry


class MeshTags:
    """Mesh tags associate data (markers) with a subset of mesh entities of a given dimension."""

    def __init__(self, meshtags):
        """Initialize tags from a C++ MeshTags object.

        Args:
            meshtags: C++ mesh tags object.

        Note:
            MeshTags objects should not usually be created using this
            initializer directly.

            A Python mesh is passed to the initializer as it may have
            UFL data attached that is not attached the C + + Mesh that is
            associated with the C + + ``meshtags`` object. If `mesh` is
            passed, ``mesh`` and ``meshtags`` must share the same C + +
            mesh.
        """
        self._cpp_object = meshtags

    def ufl_id(self) -> int:
        """Identiftying integer used by UFL."""
        return id(self)

    @property
    def topology(self) -> _cpp.mesh.Topology:
        """Mesh topology with which the the tags are associated."""
        return self._cpp_object.topology

    @property
    def dim(self) -> int:
        """Topological dimension of the tagged entities."""
        return self._cpp_object.dim

    @property
    def indices(self) -> npt.NDArray[np.int32]:
        """Indices of tagged mesh entities."""
        return self._cpp_object.indices

    @property
    def values(self):
        """Values associated with tagged mesh entities."""
        return self._cpp_object.values

    @property
    def name(self) -> str:
        "Name of the mesh tags object."
        return self._cpp_object.name

    @name.setter
    def name(self, value):
        self._cpp_object.name = value

    def find(self, value) -> npt.NDArray[np.int32]:
        """Get a list of all entity indices with a given value.

        Args:
            value: Tag value to search for.

        Returns:
            Indices of entities with tag ``value``.
        """
        return self._cpp_object.find(value)


def compute_incident_entities(
    topology: Topology, entities: npt.NDArray[np.int32], d0: int, d1: int
) -> npt.NDArray[np.int32]:
    """Compute all entities of ``d1`` connected to ``entities`` of dimension ``d0``.

    Args:
        topology: The topology
        entities: List of entities fo dimension ``d0``.
        d0: Topological dimension
        d1: Topological dimension to map to

    """
    return _cpp.mesh.compute_incident_entities(topology._cpp_object, entities, d0, d1)


def compute_midpoints(mesh: Mesh, dim: int, entities: npt.NDArray[np.int32]):
    return _cpp.mesh.compute_midpoints(mesh._cpp_object, dim, entities)


def locate_entities(mesh: Mesh, dim: int, marker: typing.Callable) -> np.ndarray:
    """Compute mesh entities satisfying a geometric marking function.

    Args:
        mesh: Mesh to locate entities on.
        dim: Topological dimension of the mesh entities to consider.
        marker: A function that takes an array of points ``x`` with
            shape ``(gdim, num_points)`` and returns an array of
            booleans of length ``num_points``, evaluating to `True` for
            entities to be located.

    Returns:
        Indices (local to the process) of marked mesh entities.
    """
    return _cpp.mesh.locate_entities(mesh._cpp_object, dim, marker)


def locate_entities_boundary(mesh: Mesh, dim: int, marker: typing.Callable) -> np.ndarray:
    """Compute mesh entities that are connected to an owned boundary
    facet and satisfy a geometric marking function.

    For vertices and edges, in parallel this function will not
    necessarily mark all entities that are on the exterior boundary. For
    example, it is possible for a process to have a vertex that lies on
    the boundary without any of the attached facets being a boundary
    facet. When used to find degrees-of-freedom, e.g. using
    :func:`dolfinx.fem.locate_dofs_topological`, the function that uses
    the data returned by this function must typically perform some
    parallel communication.

    Args:
        mesh: Mesh to locate boundary entities on.
        dim: Topological dimension of the mesh entities to consider
        marker: Function that takes an array of points ``x`` with shape
            ``(gdim, num_points)`` and returns an array of booleans of
            length ``num_points``, evaluating to ``True`` for entities
            to be located.

    Returns:
        Indices (local to the process) of marked mesh entities.
    """
    return _cpp.mesh.locate_entities_boundary(mesh._cpp_object, dim, marker)


def transfer_meshtag(
    meshtag: MeshTags,
    mesh1: Mesh,
    parent_cell: npt.NDArray[np.int32],
    parent_facet: typing.Optional[npt.NDArray[np.int8]] = None,
) -> MeshTags:
    """Generate cell mesh tags on a refined mesh from the mesh tags on the coarse parent mesh.

    Args:
        meshtag: Mesh tags on the coarse, parent mesh.
        mesh1: The refined mesh.
        parent_cell: Index of the parent cell for each cell in the
            refined mesh.
        parent_facet: Index of the local parent facet for each cell
            in the refined mesh. Only required for transfer tags on
            facets.

    Returns:
        Mesh tags on the refined mesh.
    """
    if meshtag.dim == meshtag.topology.dim:
        mt = _cpp.refinement.transfer_cell_meshtag(
            meshtag._cpp_object, mesh1.topology._cpp_object, parent_cell
        )
        return MeshTags(mt)
    elif meshtag.dim == meshtag.topology.dim - 1:
        assert parent_facet is not None
        mt = _cpp.refinement.transfer_facet_meshtag(
            meshtag._cpp_object, mesh1.topology._cpp_object, parent_cell, parent_facet
        )
        return MeshTags(mt)
    else:
        raise RuntimeError("MeshTag transfer is supported on on cells or facets.")


def refine(
    mesh: Mesh, edges: typing.Optional[np.ndarray] = None, redistribute: bool = True
) -> Mesh:
    """Refine a mesh.

    Args:
        mesh: Mesh from which to create the refined mesh.
        edges: Indices of edges to split during refinement. If ``None``,
            mesh refinement is uniform.
        redistribute:
            Refined mesh is re-partitioned if ``True``.

    Returns:
       Refined mesh.
    """
    if edges is None:
        mesh1 = _cpp.refinement.refine(mesh._cpp_object, redistribute)
    else:
        mesh1 = _cpp.refinement.refine(mesh._cpp_object, edges, redistribute)
    ufl_domain = ufl.Mesh(mesh._ufl_domain.ufl_coordinate_element())  # type: ignore
    return Mesh(mesh1, ufl_domain)


def refine_interval(
    mesh: Mesh,
    cells: typing.Optional[np.ndarray] = None,
    redistribute: bool = True,
    ghost_mode: GhostMode = GhostMode.shared_facet,
) -> tuple[Mesh, npt.NDArray[np.int32]]:
    """Refine a (topologically) one dimensional mesh.

    Args:
        mesh: Mesh to refine
        cells: Indices of cells, i.e. edges, to split druing refinement. If ``None``, mesh
            refinement is uniform.
        redistribute: Refined mesh is re-partitioned if ``True``.
        ghost_mode: ghost mode of the refined mesh

    Returns:
        Refined mesh and parent cells
    """

    if cells is None:
        refined_mesh, parent_cells = _cpp.refinement.refine_interval(
            mesh._cpp_object, redistribute, ghost_mode
        )
    else:
        refined_mesh, parent_cells = _cpp.refinement.refine_interval(
            mesh._cpp_object, cells, redistribute, ghost_mode
        )

    return Mesh(refined_mesh, mesh._ufl_domain), parent_cells


def refine_plaza(
    mesh: Mesh,
    edges: typing.Optional[np.ndarray] = None,
    redistribute: bool = True,
    option: RefinementOption = RefinementOption.none,
) -> tuple[Mesh, npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Refine a mesh.

    Args:
        mesh: Mesh from which to create the refined mesh.
        edges: Indices of edges to split during refinement. If ``None``,
            mesh refinement is uniform.
        redistribute:
            Refined mesh is re-partitioned if ``True``.
        option:
            Control computation of the parent-refined mesh data.

    Returns:
       Refined mesh, list of parent cell for each refine cell, and list
    """
    if edges is None:
        mesh1, cells, facets = _cpp.refinement.refine_plaza(mesh._cpp_object, redistribute, option)
    else:
        mesh1, cells, facets = _cpp.refinement.refine_plaza(
            mesh._cpp_object, edges, redistribute, option
        )
    return Mesh(mesh1, mesh._ufl_domain), cells, facets


def create_mesh(
    comm: _MPI.Comm,
    cells: npt.NDArray[np.int64],
    x: npt.NDArray[np.floating],
    e: typing.Union[
        ufl.Mesh,
        basix.finite_element.FiniteElement,
        basix.ufl._BasixElement,
        _CoordinateElement,
    ],
    partitioner: typing.Optional[typing.Callable] = None,
) -> Mesh:
    """Create a mesh from topology and geometry arrays.

    Args:
        comm: MPI communicator to define the mesh on.
        cells: Cells of the mesh. ``cells[i]`` are the 'nodes' of cell ``i``.
        x: Mesh geometry ('node' coordinates), with shape ``(num_nodes, gdim)``.
        e: UFL mesh. The mesh scalar type is determined by the scalar
            type of ``e``.
        partitioner: Function that computes the parallel distribution of
            cells across MPI ranks.

    Note:
        If required, the coordinates ``x`` will be cast to the same type
        as the domain/element ``e``.

    Returns:
        A mesh.
    """
    if partitioner is None and comm.size > 1:
        partitioner = _cpp.mesh.create_cell_partitioner(GhostMode.none)

    x = np.asarray(x, order="C")
    if x.ndim == 1:
        gdim = 1
    else:
        gdim = x.shape[1]

    dtype = None
    try:
        # e is a UFL domain
        e_ufl = e.ufl_coordinate_element()  # type: ignore
        cmap = _coordinate_element(e_ufl.basix_element)  # type: ignore
        domain = e
        dtype = cmap.dtype
        # TODO: Resolve UFL vs Basix geometric dimension issue
        # assert domain.geometric_dimension() == gdim
    except AttributeError:
        try:
            # e is a Basix 'UFL' element
            cmap = _coordinate_element(e.basix_element)  # type: ignore
            domain = ufl.Mesh(e)
            dtype = cmap.dtype
            assert domain.geometric_dimension() == gdim
        except AttributeError:
            try:
                # e is a Basix element
                # TODO: Resolve geometric dimension vs shape for manifolds
                cmap = _coordinate_element(e)  # type: ignore
                e_ufl = basix.ufl._BasixElement(e)  # type: ignore
                e_ufl = basix.ufl.blocked_element(e_ufl, shape=(gdim,))
                domain = ufl.Mesh(e_ufl)
                dtype = cmap.dtype
                assert domain.geometric_dimension() == gdim
            except (AttributeError, TypeError):
                # e is a CoordinateElement
                cmap = e
                domain = None
                dtype = cmap.dtype

    x = np.asarray(x, dtype=dtype, order="C")
    cells = np.asarray(cells, dtype=np.int64, order="C")
    mesh = _cpp.mesh.create_mesh(comm, cells, cmap._cpp_object, x, partitioner)

    return Mesh(mesh, domain)


def create_submesh(
    msh: Mesh, dim: int, entities: npt.NDArray[np.int32]
) -> tuple[Mesh, npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Create a mesh with specified entities from another mesh.

    Args:
        mesh: Mesh to create the sub-mesh from.
        dim: Topological dimension of the entities in ``msh`` to include in the sub-mesh.
        entities: Indices of entities in ``msh`` to include in the sub-mesh.
    Returns:
        The (1) sub mesh, (2) entity map, (3) vertex map, and (4) node map (geometry).
        Each of the maps a local index of the sub mesh to a local index of ``msh``.
    """
    submsh, entity_map, vertex_map, geom_map = _cpp.mesh.create_submesh(
        msh._cpp_object, dim, entities
    )
    submsh_ufl_cell = ufl.Cell(submsh.topology.cell_name())
    submsh_domain = ufl.Mesh(
        basix.ufl.element(
            "Lagrange",
            submsh_ufl_cell.cellname(),
            submsh.geometry.cmap.degree,
            basix.LagrangeVariant(submsh.geometry.cmap.variant),
            shape=(submsh.geometry.dim,),
            dtype=submsh.geometry.x.dtype,
        )
    )
    return (Mesh(submsh, submsh_domain), entity_map, vertex_map, geom_map)


def meshtags(
    mesh: Mesh,
    dim: int,
    entities: npt.NDArray[np.int32],
    values: typing.Union[np.ndarray, int, float],
) -> MeshTags:
    """Create a MeshTags object that associates data with a subset of mesh entities.

    Args:
        mesh: The mesh.
        dim: Topological dimension of the mesh entity.
        entities: Indices(local to process) of entities to associate
            values with . The array must be sorted and must not contain
            duplicates.
        values: The corresponding value for each entity.

    Returns:
        A mesh tags object.

    Note:
        The type of the returned MeshTags is inferred from the type of
        ``values``.

    """

    if isinstance(values, int):
        assert values >= np.iinfo(np.int32).min and values <= np.iinfo(np.int32).max
        values = np.full(entities.shape, values, dtype=np.int32)
    elif isinstance(values, float):
        values = np.full(entities.shape, values, dtype=np.double)

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

    return MeshTags(
        ftype(mesh.topology._cpp_object, dim, np.asarray(entities, dtype=np.int32), values)
    )


def meshtags_from_entities(
    mesh: Mesh, dim: int, entities: _cpp.graph.AdjacencyList_int32, values: npt.NDArray[typing.Any]
):
    """Create a :class:dolfinx.mesh.MeshTags` object that associates
    data with a subset of mesh entities, where the entities are defined
    by their vertices.

    Args:
        mesh: The mesh.
        dim: Topological dimension of the mesh entity.
        entities: Entities to associated values with, with entities
            defined by their vertices.
        values: The corresponding value for each entity.

    Returns:
        A mesh tags object.

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
    return MeshTags(_cpp.mesh.create_meshtags(mesh.topology._cpp_object, dim, entities, values))


def create_interval(
    comm: _MPI.Comm,
    nx: int,
    points: npt.ArrayLike,
    dtype: npt.DTypeLike = default_real_type,
    ghost_mode=GhostMode.shared_facet,
    partitioner=None,
) -> Mesh:
    """Create an interval mesh.

    Args:
        comm: MPI communicator.
        nx: Number of cells.
        points: Coordinates of the end points.
        dtype: Float type for the mesh geometry(``numpy.float32``
            or ``numpy.float64``).
        ghost_mode: Ghost mode used in the mesh partitioning. Options
            are ``GhostMode.none`` and ``GhostMode.shared_facet``.
        partitioner: Partitioning function to use for determining the
            parallel distribution of cells across MPI ranks.

    Returns:
        An interval mesh.

    """
    if partitioner is None and comm.size > 1:
        partitioner = _cpp.mesh.create_cell_partitioner(ghost_mode)
    domain = ufl.Mesh(basix.ufl.element("Lagrange", "interval", 1, shape=(1,), dtype=dtype))  # type: ignore
    if np.issubdtype(dtype, np.float32):
        mesh = _cpp.mesh.create_interval_float32(comm, nx, points, ghost_mode, partitioner)
    elif np.issubdtype(dtype, np.float64):
        mesh = _cpp.mesh.create_interval_float64(comm, nx, points, ghost_mode, partitioner)
    else:
        raise RuntimeError(f"Unsupported mesh geometry float type: {dtype}")
    return Mesh(mesh, domain)


def create_unit_interval(
    comm: _MPI.Comm,
    nx: int,
    dtype: npt.DTypeLike = default_real_type,
    ghost_mode=GhostMode.shared_facet,
    partitioner=None,
) -> Mesh:
    """Create a mesh on the unit interval.

    Args:
        comm: MPI communicator.
        nx: Number of cells.
        points: Coordinates of the end points.
        dtype: Float type for the mesh geometry(``numpy.float32``
            or ``numpy.float64``).
        ghost_mode: Ghost mode used in the mesh partitioning. Options
            are ``GhostMode.none`` and ``GhostMode.shared_facet``.
        partitioner: Partitioning function to use for determining the
            parallel distribution of cells across MPI ranks.

    Returns:
        A unit interval mesh with end points at 0 and 1.
    """
    return create_interval(comm, nx, [0.0, 1.0], dtype, ghost_mode, partitioner)


def create_rectangle(
    comm: _MPI.Comm,
    points: npt.ArrayLike,
    n: npt.ArrayLike,
    cell_type=CellType.triangle,
    dtype: npt.DTypeLike = default_real_type,
    ghost_mode=GhostMode.shared_facet,
    partitioner=None,
    diagonal: DiagonalType = DiagonalType.right,
) -> Mesh:
    """Create a rectangle mesh.

    Args:
        comm: MPI communicator.
        points: Coordinates of the lower - left and upper - right corners of
            the rectangle.
        n: Number of cells in each direction.
        cell_type: Mesh cell type.
        dtype: Float type for the mesh geometry(``numpy.float32``
            or ``numpy.float64``)
        ghost_mode: Ghost mode used in the mesh partitioning.
        partitioner: Function that computes the parallel distribution of
            cells across MPI ranks.
        diagonal: Direction of diagonal of triangular meshes. The
            options are ``left``, ``right``, ``crossed``, ``left / right``,
            ``right / left``.

    Returns:
        A mesh of a rectangle.
    """
    if partitioner is None and comm.size > 1:
        partitioner = _cpp.mesh.create_cell_partitioner(ghost_mode)
    domain = ufl.Mesh(basix.ufl.element("Lagrange", cell_type.name, 1, shape=(2,), dtype=dtype))  # type: ignore
    if np.issubdtype(dtype, np.float32):
        mesh = _cpp.mesh.create_rectangle_float32(comm, points, n, cell_type, partitioner, diagonal)
    elif np.issubdtype(dtype, np.float64):
        mesh = _cpp.mesh.create_rectangle_float64(comm, points, n, cell_type, partitioner, diagonal)
    else:
        raise RuntimeError(f"Unsupported mesh geometry float type: {dtype}")
    return Mesh(mesh, domain)


def create_unit_square(
    comm: _MPI.Comm,
    nx: int,
    ny: int,
    cell_type=CellType.triangle,
    dtype: npt.DTypeLike = default_real_type,
    ghost_mode=GhostMode.shared_facet,
    partitioner=None,
    diagonal: DiagonalType = DiagonalType.right,
) -> Mesh:
    """Create a mesh of a unit square.

    Args:
        comm: MPI communicator.
        nx: Number of cells in the "x" direction.
        ny: Number of cells in the "y" direction.
        cell_type: Mesh cell type.
        dtype: Float type for the mesh geometry(``numpy.float32``
            or ``numpy.float64``).
        ghost_mode: Ghost mode used in the mesh partitioning.
        partitioner: Function that computes the parallel distribution of
            cells across MPI ranks.
        diagonal:
            Direction of diagonal.

    Returns:
        A mesh of a square with corners at (0, 0) and (1, 1).
    """
    return create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [nx, ny],
        cell_type,
        dtype,
        ghost_mode,
        partitioner,
        diagonal,
    )


def create_box(
    comm: _MPI.Comm,
    points: list[npt.ArrayLike],
    n: list,
    cell_type=CellType.tetrahedron,
    dtype: npt.DTypeLike = default_real_type,
    ghost_mode=GhostMode.shared_facet,
    partitioner=None,
) -> Mesh:
    """Create a box mesh.

    Args:
        comm: MPI communicator.
        points: Coordinates of the 'lower-left' and 'upper-right'
            corners of the box.
        n: List of cells in each direction
        cell_type: The cell type.
        dtype: Float type for the mesh geometry(``numpy.float32``
            or ``numpy.float64``).
        ghost_mode: The ghost mode used in the mesh partitioning.
        partitioner: Function that computes the parallel distribution of
            cells across MPI ranks.

    Returns:
        A mesh of a box domain.
    """
    if partitioner is None and comm.size > 1:
        partitioner = _cpp.mesh.create_cell_partitioner(ghost_mode)
    domain = ufl.Mesh(basix.ufl.element("Lagrange", cell_type.name, 1, shape=(3,), dtype=dtype))  # type: ignore
    if np.issubdtype(dtype, np.float32):
        mesh = _cpp.mesh.create_box_float32(comm, points, n, cell_type, partitioner)
    elif np.issubdtype(dtype, np.float64):
        mesh = _cpp.mesh.create_box_float64(comm, points, n, cell_type, partitioner)
    else:
        raise RuntimeError(f"Unsupported mesh geometry float type: {dtype}")
    return Mesh(mesh, domain)


def create_unit_cube(
    comm: _MPI.Comm,
    nx: int,
    ny: int,
    nz: int,
    cell_type=CellType.tetrahedron,
    dtype: npt.DTypeLike = default_real_type,
    ghost_mode=GhostMode.shared_facet,
    partitioner=None,
) -> Mesh:
    """Create a mesh of a unit cube.

    Args:
        comm: MPI communicator.
        nx: Number of cells in "x" direction.
        ny: Number of cells in "y" direction.
        nz: Number of cells in "z" direction.
        cell_type: Mesh cell type
        dtype: Float type for the mesh geometry(``numpy.float32``
            or ``numpy.float64``).
        ghost_mode: Ghost mode used in the mesh partitioning.
        partitioner: Function that computes the parallel distribution of
            cells across MPI ranks.

    Returns:
        A mesh of an axis-aligned unit cube with corners at ``(0, 0, 0)``
            and ``(1, 1, 1)``.
    """
    return create_box(
        comm,
        [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])],
        [nx, ny, nz],
        cell_type,
        dtype,
        ghost_mode,
        partitioner,
    )


def entities_to_geometry(
    mesh: Mesh, dim: int, entities: npt.NDArray[np.int32], permute=False
) -> npt.NDArray[np.int32]:
    """Compute the geometric DOFs associated with the closure of the given mesh entities.

    Args:
        mesh: The mesh.
        dim: Topological dimension of the entities of interest.
        entities: Entity indices (local to the process).
        permute: Permute the DOFs such that they are consistent with the orientation
            of `dim`-dimensional mesh entities. This requires `create_entity_permutations` to
            be called first.

    Returns:
        The geometric DOFs associated with the closure of the entities in `entities`.
    """
    return _cpp.mesh.entities_to_geometry(mesh._cpp_object, dim, entities, permute)


def exterior_facet_indices(topology: Topology) -> npt.NDArray[np.int32]:
    """Compute the indices of all exterior facets that are owned by the caller.

    An exterior facet (co-dimension 1) is one that is connected globally to
    only one cell of co-dimension 0).

    Note:
        This is a collective operation that should be called on all processes.

    Args:
        topology: The topology

    Returns:
        Sorted list of owned facet indices that are exterior facets of the mesh.
    """
    return _cpp.mesh.exterior_facet_indices(topology._cpp_object)
