# Copyright (C) 2017-2024 Chris N. Richardson, Garth N. Wells and
# Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Creation, refining and marking of meshes."""

from __future__ import annotations

import typing
from collections.abc import Callable, Sequence

from mpi4py import MPI as _MPI

import numpy as np
import numpy.typing as npt

import basix
import basix.ufl
import ufl
from dolfinx import cpp as _cpp
from dolfinx import default_real_type
from dolfinx.common import IndexMap as _IndexMap
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
from dolfinx.cpp.refinement import IdentityPartitionerPlaceholder, RefinementOption
from dolfinx.fem import CoordinateElement as _CoordinateElement
from dolfinx.fem import coordinate_element as _coordinate_element
from dolfinx.graph import AdjacencyList

__all__ = [
    "CellType",
    "EntityMap",
    "Geometry",
    "GhostMode",
    "Mesh",
    "MeshTags",
    "Topology",
    "build_dual_graph",
    "cell_dim",
    "compute_incident_entities",
    "compute_midpoints",
    "create_box",
    "create_cell_partitioner",
    "create_geometry",
    "create_interval",
    "create_mesh",
    "create_rectangle",
    "create_submesh",
    "create_unit_cube",
    "create_unit_interval",
    "create_unit_square",
    "entities_to_geometry",
    "exterior_facet_indices",
    "locate_entities",
    "locate_entities_boundary",
    "meshtags",
    "meshtags_from_entities",
    "refine",
    "to_string",
    "to_type",
    "transfer_meshtag",
]


class Topology:
    """Topology for a :class:`dolfinx.mesh.Mesh`"""

    _cpp_object: _cpp.mesh.Topology

    def __init__(self, topology: _cpp.mesh.Topology):
        """Initialize a topology from a C++ topology.
        Args:
            topology: The C++ topology object

        Note:
            Topology objects should usually be constructed with the
            :func:`dolfinx.cpp.mesh.create_topology` and not this class
            initializer.
        """
        self._cpp_object = topology

    def cell_name(self) -> str:
        """String representation of the cell-type of the topology"""
        return to_string(self._cpp_object.cell_type)

    def connectivity(self, d0: int, d1: int) -> _cpp.graph.AdjacencyList_int32:
        """Return connectivity from entities of dimension ``d0`` to
        entities of dimension ``d1``.

        Args:
            d0: Dimension of entity one is mapping from
            d1: Dimension of entity one is mapping to
        """
        if (conn := self._cpp_object.connectivity(d0, d1)) is not None:
            return conn
        else:
            raise RuntimeError(
                f"Connectivity between dimension {d0} and {d1} has not been computed.",
                f"Please call `dolfinx.mesh.Topology.create_connectivity({d0}, {d1})` first.",
            )

    @property
    def comm(self):
        return self._cpp_object.comm

    def create_connectivity(self, d0: int, d1: int):
        """Build entity connectivity ``d0 -> d1``.

        Args:
            d0: Dimension of entities connectivity is from.
            d1: Dimension of entities connectivity is to.
        """
        self._cpp_object.create_connectivity(d0, d1)

    def create_entities(self, dim: int) -> bool:
        """Create entities of given topological dimension.

        Args:
            dim: Topological dimension of entities to create.

        Returns:
            ``True` is entities are created, ``False`` is if entities
            already existed.
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
        """Entity types in the topology for all topological dimensions."""
        return self._cpp_object.entity_types

    def get_cell_permutation_info(self) -> npt.NDArray[np.uint32]:
        """Get permutation information.

        The returned data is used for packing coefficients and
        assembling of tensors. The bits of each integer encodes the
        number of reflections and permutations for each sub-entity of
        the cell to be able to map it to the reference element.
        """
        return self._cpp_object.get_cell_permutation_info()

    def get_facet_permutations(self) -> npt.NDArray[np.uint8]:
        """Get the permutation integer to apply to facets.

        The bits of each integer describes the number of reflections and
        rotations that has to be applied to a facet to map between a
        facet in the mesh (relative to a cell) and the corresponding
        facet on the reference element. The data has the shape
        ``(num_cells, num_facets_per_cell)``, flattened row-wise. The
        number of cells include potential ghost cells.

        Note:
            The data can be unpacked with ``numpy.unpack_bits``.
        """
        return self._cpp_object.get_facet_permutations()

    def index_map(self, dim: int) -> _cpp.common.IndexMap:
        """Get the IndexMap that describes the parallel distribution of the
        mesh entities.

        Args:
            dim: Topological dimension.

        Returns:
            Index map for the entities of dimension ``dim``.
        """
        if (imap := self._cpp_object.index_map(dim)) is not None:
            return imap
        else:
            raise RuntimeError(
                f"Entities of dimension {dim} has not been computed."
                f"Call `dolfinx.mesh.Topology.create_entities({dim}) first."
            )

    def index_maps(self, dim: int) -> list[_cpp.common.IndexMap]:
        """Get the IndexMaps that describes the parallel distribution of
           the mesh entities, for each entity type of the dimension.

        Args:
            dim: Topological dimension.

        Returns:
            List of IndexMaps for the entities of dimension ``dim``.
        """
        if (imaps := self._cpp_object.index_maps(dim)) is not None:
            return imaps
        else:
            raise RuntimeError(
                f"Entities of dimension {dim} have not been computed."
                f"Call `dolfinx.mesh.Topology.create_entities({dim}) first."
            )

    def interprocess_facets(self) -> npt.NDArray[np.int32]:
        """List of inter-process facets, if facet topology has been
        computed."""
        return self._cpp_object.interprocess_facets()

    @property
    def original_cell_index(self) -> npt.NDArray[np.int64]:
        """Get the original cell index"""
        return self._cpp_object.original_cell_index

    @property
    def cell_type(self) -> CellType:
        """Get the cell type of the topology."""
        return self._cpp_object.cell_type


class Geometry:
    """The geometry of a :class:`dolfinx.mesh.Mesh`"""

    _cpp_object: typing.Union[_cpp.mesh.Geometry_float32, _cpp.mesh.Geometry_float64]

    def __init__(
        self, geometry: typing.Union[_cpp.mesh.Geometry_float32, _cpp.mesh.Geometry_float64]
    ):
        """Initialize a geometry from a C++ geometry.

        Args:
            geometry: The C++ geometry object.

        Note:
            Geometry objects should usually be constructed with the
            :func:`create_geometry` and not using this class
            initializer. This class is combined with different base
            classes that depend on the scalar type used in the Geometry.
        """
        self._cpp_object = geometry

    @property
    def cmap(self) -> _CoordinateElement:
        """Element that describes the geometry map."""
        return _CoordinateElement(self._cpp_object.cmap)

    @property
    def dim(self):
        """Dimension of the Euclidean coordinate system."""
        return self._cpp_object.dim

    @property
    def dofmap(self) -> npt.NDArray[np.int32]:
        """Dofmap for the geometry, shape
        ``(num_cells, dofs_per_cell)``."""
        return self._cpp_object.dofmap

    def index_map(self) -> _IndexMap:
        """Index map describing the layout of the geometry points
        (nodes)."""
        return self._cpp_object.index_map()

    @property
    def input_global_indices(self) -> npt.NDArray[np.int64]:
        """Global input indices of the geometry nodes."""
        return self._cpp_object.input_global_indices

    @property
    def x(self) -> typing.Union[npt.NDArray[np.float32], npt.NDArray[np.float64]]:
        """Geometry coordinate points,  ``shape=(num_points, 3)``."""
        return self._cpp_object.x


class Mesh:
    """A mesh."""

    _mesh: typing.Union[_cpp.mesh.Mesh_float32, _cpp.mesh.Mesh_float64]
    _topology: Topology
    _geometry: Geometry
    _ufl_domain: typing.Optional[ufl.Mesh]

    def __init__(
        self,
        msh: typing.Union[_cpp.mesh.Mesh_float32, _cpp.mesh.Mesh_float64],
        domain: typing.Optional[ufl.Mesh],
    ):
        """Initialize mesh from a C++ mesh.

        Args:
            msh: A C++ mesh object.
            domain: A UFL domain.

        Note:
            Mesh objects should usually be constructed using
            :func:`create_mesh` and not using this class initializer.
            This class is combined with different base classes that
            depend on the scalar type used in the Mesh.
        """
        self._cpp_object = msh
        self._topology = Topology(self._cpp_object.topology)
        self._geometry = Geometry(self._cpp_object.geometry)
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

    def ufl_domain(self) -> typing.Optional[ufl.Mesh]:
        """Return the ufl domain corresponding to the mesh.

        Returns:
            The UFL domain. Is ``None`` if the domain has not been set.

        Note:
            This method is required for UFL compatibility.
        """
        return self._ufl_domain

    def basix_cell(self) -> basix.CellType:
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
    def geometry(self) -> Geometry:
        "Mesh geometry."
        return self._geometry


class MeshTags:
    """Mesh tags associate data (markers) with a subset of mesh entities of
    a given dimension."""

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
        """Mesh topology with which the tags are associated."""
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


class EntityMap:
    """A bidirectional map that relates entities in two different
    topologies.
    """

    def __init__(self, entity_map):
        """Initialise an entity map from a C++ `EntityMap` object.

        Args:
            entity_map: A C++ `EntityMap` object

        .. note::

            `EntityMap` objects should not usually be created using this
            initializer directly.
        """
        self._cpp_object = entity_map
        self._topology = Topology(self._cpp_object.topology)
        self._sub_topology = Topology(self._cpp_object.sub_topology)

    def sub_topology_to_topology(self, entities, inverse):
        """Map entities between the sub-topology and the parent topology.

        If `inverse` is False, this function maps a list of
        `self.dim()`-dimensional entities from `self.sub_topology()` to
        the corresponding entities in `self.topology()`. If `inverse` is
        True, it performs the inverse mapping from `self.topology()` to
        `self.sub_topology()`. Entities that do not exist in the
        sub-topology are marked as -1.

        Note:
            If `inverse` is `True`, this function recomputes the inverse
            map on every call (it is not cached), which may be expensive
            if called repeatedly.

        Args:
            entities:
                A list of entity indices in the source topology.
            inverse:
                If False, maps from `self.sub_topology()` to
                `self.topology()`. If True, maps from `this.topology()`
                to `this.sub_topology()`.

        Returns:
            A list of mapped entity indices. Entities that don't exist
            in the target topology are marked as -1.
        """
        return self._cpp_object.sub_topology_to_topology(entities, inverse)

    @property
    def dim(self):
        """Get the topological dimension of the entities related by this
        EntityMap.

        Returns:
            int: The topological dimension
        """
        return self._cpp_object.dim

    @property
    def topology(self):
        return self._topology

    @property
    def sub_topology(self):
        return self._sub_topology


def entity_map(topology, sub_topology, dim, sub_topology_to_topology):
    """Create a bidirectional map relating entities of dimension `dim` in
    `topology` and `sub_topology`.

    Args:
        topology: A topology
        sub_topology: Topology of another mesh. This must be a
            "sub-topology" of `topology` i.e. every entity in
            `sub_topology` must also exist in `topology`.
        dim: The dimension of the entities
        sub_topology_to_topology: A list of entities in `topology` where
            `sub_topology_to_topology[i]` is the index in `topology`
            corresponding to entity `i` in `sub_topology`.
    """
    return _cpp.mesh.EntityMap(
        topology._cpp_object, sub_topology._cpp_object, dim, sub_topology_to_topology
    )


def compute_incident_entities(
    topology: Topology, entities: npt.NDArray[np.int32], d0: int, d1: int
) -> npt.NDArray[np.int32]:
    """Compute all entities of ``d1`` connected to ``entities`` of
    dimension ``d0``.

    Args:
        topology: The topology.
        entities: List of entities of dimension ``d0``.
        d0: Topological dimension.
        d1: Topological dimension to map to.

    Returns:
        Incident entity indices.
    """
    return _cpp.mesh.compute_incident_entities(topology._cpp_object, entities, d0, d1)


def compute_midpoints(msh: Mesh, dim: int, entities: npt.NDArray[np.int32]):
    """Compute the midpoints of a set of mesh entities.

    Args:
        msh: The mesh.
        dim: Topological dimension of the mesh entities to consider.
        entities: Indices of entities in ``mesh`` to consider.

    Returns:
        Midpoints of the entities, shape ``(num_entities, 3)``.
    """
    return _cpp.mesh.compute_midpoints(msh._cpp_object, dim, entities)


def locate_entities(msh: Mesh, dim: int, marker: Callable) -> np.ndarray:
    """Compute mesh entities satisfying a geometric marking function.

    Args:
        msh: Mesh to locate entities on.
        dim: Topological dimension of the mesh entities to consider.
        marker: A function that takes an array of points ``x`` with
            shape ``(gdim, num_points)`` and returns an array of
            booleans of length ``num_points``, evaluating to `True` for
            entities to be located.

    Returns:
        Indices (local to the process) of marked mesh entities.
    """
    return _cpp.mesh.locate_entities(msh._cpp_object, dim, marker)


def locate_entities_boundary(msh: Mesh, dim: int, marker: Callable) -> np.ndarray:
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
        msh: Mesh to locate boundary entities on.
        dim: Topological dimension of the mesh entities to consider
        marker: Function that takes an array of points ``x`` with shape
            ``(gdim, num_points)`` and returns an array of booleans of
            length ``num_points``, evaluating to ``True`` for entities
            to be located.

    Returns:
        Indices (local to the process) of marked mesh entities.
    """
    return _cpp.mesh.locate_entities_boundary(msh._cpp_object, dim, marker)


def transfer_meshtag(
    meshtag: MeshTags,
    msh1: Mesh,
    parent_cell: npt.NDArray[np.int32],
    parent_facet: typing.Optional[npt.NDArray[np.int8]] = None,
) -> MeshTags:
    """Generate cell mesh tags on a refined mesh from the mesh tags on the
    coarse parent mesh.

    Args:
        meshtag: Mesh tags on the coarse, parent mesh.
        msh1: The refined mesh.
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
            meshtag._cpp_object, msh1.topology._cpp_object, parent_cell
        )
        return MeshTags(mt)
    elif meshtag.dim == meshtag.topology.dim - 1:
        assert parent_facet is not None
        mt = _cpp.refinement.transfer_facet_meshtag(
            meshtag._cpp_object, msh1.topology._cpp_object, parent_cell, parent_facet
        )
        return MeshTags(mt)
    else:
        raise RuntimeError("MeshTag transfer is supported on on cells or facets.")


def refine(
    msh: Mesh,
    edges: typing.Optional[np.ndarray] = None,
    partitioner: typing.Union[
        Callable, IdentityPartitionerPlaceholder
    ] = IdentityPartitionerPlaceholder(),
    option: RefinementOption = RefinementOption.parent_cell,
) -> tuple[Mesh, npt.NDArray[np.int32], npt.NDArray[np.int8]]:
    """Refine a mesh.

    Passing ``None`` for ``partitioner``, refined cells will be on the
    same process as the parent cell.

    Note:
        Using the default partitioner for the refined mesh, the refined
        mesh will **not** include ghosts cells (cells connected by facet
        to an owned cells) even if the parent mesh is ghosted.

    Args:
        msh: Mesh from which to create the refined mesh.
        edges: Indices of edges to split during refinement. If ``None``,
            mesh refinement is uniform.
        partitioner: Partitioner to distribute the refined mesh. If a
            ``IdentityPartitionerPlaceholder`` is passed (default) no
            redistribution is performed, i.e. refined cells remain on the
            same process as the parent cell, but the ghost layer is
            updated. If a custom partitioner is passed, it will be used for
            distributing the refined mesh. If ``None`` is passed no
            redistribution will happen.
        option: Controls whether parent cells and/or parent facets are
            computed.

    Returns:
       Refined mesh, (optional) parent cells, (optional) parent facets
    """
    mesh1, parent_cell, parent_facet = _cpp.refinement.refine(
        msh._cpp_object, edges, partitioner, option
    )
    # Create new ufl domain as it will carry a reference to the C++ mesh
    # in the ufl_cargo
    ufl_domain = ufl.Mesh(msh._ufl_domain.ufl_coordinate_element())  # type: ignore
    return Mesh(mesh1, ufl_domain), parent_cell, parent_facet


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
    partitioner: typing.Optional[Callable] = None,
) -> Mesh:
    """Create a mesh from topology and geometry arrays.

    Args:
        comm: MPI communicator to define the mesh on.
            cells: Cells of the mesh. ``cells[i]`` are the 'nodes' of cell
            ``i``.
        x: Mesh geometry ('node' coordinates), with shape
            ``(num_nodes, gdim)``.
        e: UFL mesh. The mesh scalar type is determined by the scalar
            type of ``e``.
        partitioner: Function that determines the parallel distribution of
            cells across MPI ranks.

    Note:
        If required, the coordinates ``x`` will be cast to the same
        scalar type as the domain/element ``e``.

    Returns:
        A mesh.
    """
    if partitioner is None and comm.size > 1:
        partitioner = create_cell_partitioner(GhostMode.none)

    x = np.asarray(x, order="C")
    if x.ndim == 1:
        gdim = 1
    else:
        gdim = x.shape[1]

    dtype = None
    if isinstance(e, ufl.domain.Mesh):
        # e is a UFL domain
        e_ufl = e.ufl_coordinate_element()  # type: ignore
        cmap = _coordinate_element(e_ufl.basix_element)  # type: ignore
        domain = e
        dtype = cmap.dtype
        # TODO: Resolve UFL vs Basix geometric dimension issue
        # assert domain.geometric_dimension() == gdim
    elif isinstance(e, basix.finite_element.FiniteElement):
        # e is a Basix element
        # TODO: Resolve geometric dimension vs shape for manifolds
        cmap = _coordinate_element(e)  # type: ignore
        e_ufl = basix.ufl._BasixElement(e)  # type: ignore
        e_ufl = basix.ufl.blocked_element(e_ufl, shape=(gdim,))
        domain = ufl.Mesh(e_ufl)
        dtype = cmap.dtype
        assert domain.geometric_dimension() == gdim
    elif isinstance(e, ufl.finiteelement.AbstractFiniteElement):
        # e is a Basix 'UFL' element
        cmap = _coordinate_element(e.basix_element)  # type: ignore
        domain = ufl.Mesh(e)
        dtype = cmap.dtype
        assert domain.geometric_dimension() == gdim
    elif isinstance(e, _CoordinateElement):
        # e is a CoordinateElement
        cmap = e
        domain = None
        dtype = cmap.dtype  # type: ignore
    else:
        raise ValueError(f"Unsupported element type {type(e)}.")

    x = np.asarray(x, dtype=dtype, order="C")
    cells = np.asarray(cells, dtype=np.int64, order="C")
    msh: typing.Union[_cpp.mesh.Mesh_float32, _cpp.mesh.Mesh_float64] = _cpp.mesh.create_mesh(
        comm, cells, cmap._cpp_object, x, partitioner
    )

    return Mesh(msh, domain)  # type: ignore


def create_submesh(
    msh: Mesh, dim: int, entities: npt.NDArray[np.int32]
) -> tuple[Mesh, EntityMap, EntityMap, npt.NDArray[np.int32]]:
    """Create a mesh with specified entities from another mesh.

    Args:
        msh: Mesh to create the sub-mesh from.
        dim: Topological dimension of the entities in ``msh`` to include
            in the sub-mesh.
        entities: Indices of entities in ``msh`` to include in the
            sub-mesh.

    Returns:
        The (1) sub mesh, (2) entity map, (3) vertex map, and (4) node
        map (geometry). Each of the maps a local index of the sub mesh
        to a local index of ``msh``.
    """
    submsh, entity_map, vertex_map, geom_map = _cpp.mesh.create_submesh(
        msh._cpp_object, dim, entities
    )
    submsh_domain = ufl.Mesh(
        basix.ufl.element(
            "Lagrange",
            to_string(submsh.topology.cell_type),
            submsh.geometry.cmap.degree,
            basix.LagrangeVariant(submsh.geometry.cmap.variant),
            shape=(submsh.geometry.dim,),
            dtype=submsh.geometry.x.dtype,
        )
    )
    return (Mesh(submsh, submsh_domain), EntityMap(entity_map), EntityMap(vertex_map), geom_map)


def meshtags(
    msh: Mesh,
    dim: int,
    entities: npt.NDArray[np.int32],
    values: typing.Union[np.ndarray, int, float],
) -> MeshTags:
    """Create a MeshTags object that associates data with a subset of
    mesh entities.

    Args:
        msh: The mesh.
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
        ftype(msh.topology._cpp_object, dim, np.asarray(entities, dtype=np.int32), values)
    )


def meshtags_from_entities(
    msh: Mesh, dim: int, entities: AdjacencyList, values: npt.NDArray[typing.Any]
):
    """Create a :class:dolfinx.mesh.MeshTags` object that associates
    data with a subset of mesh entities, where the entities are defined
    by their vertices.

    Args:
        msh: The mesh.
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
    return MeshTags(
        _cpp.mesh.create_meshtags(msh.topology._cpp_object, dim, entities._cpp_object, values)
    )


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
    domain = ufl.Mesh(
        basix.ufl.element(
            "Lagrange",
            "interval",
            1,
            lagrange_variant=basix.LagrangeVariant.unset,
            shape=(1,),
            dtype=dtype,
        )
    )  # type: ignore
    if np.issubdtype(dtype, np.float32):
        msh = _cpp.mesh.create_interval_float32(comm, nx, points, ghost_mode, partitioner)
    elif np.issubdtype(dtype, np.float64):
        msh = _cpp.mesh.create_interval_float64(comm, nx, points, ghost_mode, partitioner)
    else:
        raise RuntimeError(f"Unsupported mesh geometry float type: {dtype}")

    return Mesh(msh, domain)


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
    n: Sequence[int],
    cell_type=CellType.triangle,
    dtype: npt.DTypeLike = default_real_type,
    ghost_mode=GhostMode.shared_facet,
    partitioner=None,
    diagonal: DiagonalType = DiagonalType.right,
) -> Mesh:
    """Create a rectangle mesh.

    Args:
        comm: MPI communicator.
        points: Coordinates of the lower - left and upper - right
            corners of the rectangle.
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
    domain = ufl.Mesh(
        basix.ufl.element(
            "Lagrange",
            cell_type.name,
            1,
            lagrange_variant=basix.LagrangeVariant.unset,
            shape=(2,),
            dtype=dtype,
        )
    )  # type: ignore
    if np.issubdtype(dtype, np.float32):
        msh = _cpp.mesh.create_rectangle_float32(comm, points, n, cell_type, partitioner, diagonal)
    elif np.issubdtype(dtype, np.float64):
        msh = _cpp.mesh.create_rectangle_float64(comm, points, n, cell_type, partitioner, diagonal)
    else:
        raise RuntimeError(f"Unsupported mesh geometry float type: {dtype}")

    return Mesh(msh, domain)


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
        A mesh of a square with corners at ``(0, 0)`` and ``(1, 1)``.
    """
    return create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        (nx, ny),
        cell_type,
        dtype,
        ghost_mode,
        partitioner,
        diagonal,
    )


def create_box(
    comm: _MPI.Comm,
    points: list[npt.ArrayLike],
    n: Sequence[int],
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
    domain = ufl.Mesh(
        basix.ufl.element(
            "Lagrange",
            cell_type.name,
            1,
            lagrange_variant=basix.LagrangeVariant.unset,
            shape=(3,),
            dtype=dtype,
        )
    )  # type: ignore
    if np.issubdtype(dtype, np.float32):
        msh = _cpp.mesh.create_box_float32(comm, points, n, cell_type, partitioner)
    elif np.issubdtype(dtype, np.float64):
        msh = _cpp.mesh.create_box_float64(comm, points, n, cell_type, partitioner)
    else:
        raise RuntimeError(f"Unsupported mesh geometry float type: {dtype}")

    return Mesh(msh, domain)


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
        (nx, ny, nz),
        cell_type,
        dtype,
        ghost_mode,
        partitioner,
    )


def entities_to_geometry(
    msh: Mesh, dim: int, entities: npt.NDArray[np.int32], permute=False
) -> npt.NDArray[np.int32]:
    """Compute the geometric DOFs associated with the closure of the
    given mesh entities.

    Args:
        msh: The mesh.
        dim: Topological dimension of the entities of interest.
        entities: Entity indices (local to the process).
        permute: Permute the DOFs such that they are consistent with the
            orientation of `dim`-dimensional mesh entities. This
            requires `create_entity_permutations` to be called first.

    Returns:
        The geometric DOFs associated with the closure of the entities
        in `entities`.
    """
    return _cpp.mesh.entities_to_geometry(msh._cpp_object, dim, entities, permute)


def exterior_facet_indices(topology: Topology) -> npt.NDArray[np.int32]:
    """Compute the indices of all exterior facets that are owned by the
    caller.

    An exterior facet (co-dimension 1) is one that is connected globally
    to only one cell of co-dimension 0).

    Note:
        This is a collective operation that should be called on all
        processes.

    Args:
        topology: The topology

    Returns:
        Sorted list of owned facet indices that are exterior facets of
        the mesh.
    """
    return _cpp.mesh.exterior_facet_indices(topology._cpp_object)


def create_geometry(
    index_map: _IndexMap,
    dofmap: npt.NDArray[np.int32],
    element: _CoordinateElement,
    x: np.ndarray,
    input_global_indices: npt.NDArray[np.int64],
) -> Geometry:
    """Create a Geometry object.

    Args:
        index_map: Index map describing the layout of the geometry
            points (nodes).
        dofmap: The geometry (point) dofmap. For a cell, it gives the
            row in the point coordinates ``x`` of each local geometry
            node. ``shape=(num_cells, num_dofs_per_cell)``.
        element: Element that describes the cell geometry map.
        x: The point coordinates. The shape is
            ``(num_points, geometric_dimension).``
        input_global_indices: The 'global' input index of each point,
            commonly from a mesh input file.
    """
    if x.dtype == np.float64:
        ftype = _cpp.mesh.Geometry_float64
    elif x.dtype == np.float32:
        ftype = _cpp.mesh.Geometry_float64
    else:
        raise ValueError("Unknown floating type for geometry, got: {x.dtype}")

    if (dtype := np.dtype(element.dtype)) != x.dtype:
        raise ValueError(f"Mismatch in x dtype ({x.dtype}) and coordinate element ({dtype})")

    return Geometry(ftype(index_map, dofmap, element, x, input_global_indices))
