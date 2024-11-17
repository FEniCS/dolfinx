# Copyright (C) 2018-2021 Michal Habera, Garth N. Wells and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Methods for geometric searches and operations."""

from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt

if typing.TYPE_CHECKING:
    from dolfinx.mesh import Mesh

from dolfinx import cpp as _cpp
from dolfinx.graph import AdjacencyList

__all__ = [
    "BoundingBoxTree",
    "bb_tree",
    "compute_colliding_cells",
    "squared_distance",
    "compute_closest_entity",
    "compute_collisions_trees",
    "compute_collisions_points",
    "compute_distance_gjk",
    "create_midpoint_tree",
    "PointOwnershipData",
]


class PointOwnershipData:
    """Convenience class for storing data related to the ownership of points."""

    _cpp_object: typing.Union[
        _cpp.geometry.PointOwnershipData_float32, _cpp.geometry.PointOwnershipData_float64
    ]

    def __init__(self, ownership_data):
        """Wrap a C++ PointOwnershipData."""
        self._cpp_object = ownership_data

    def src_owner(self) -> npt.NDArray[np.int32]:
        """Ranks owning each point sent into ownership determination for current process."""
        return self._cpp_object.src_owner

    def dest_owner(self) -> npt.NDArray[np.int32]:
        """Ranks that sent ``dest_points`` to current process."""
        return self._cpp_object.dest_owners

    def dest_points(self) -> npt.NDArray[np.floating]:
        """Points owned by current rank."""
        return self._cpp_object.dest_points

    def dest_cells(self) -> npt.NDArray[np.int32]:
        """Cell indices (local to process) where each entry of ``dest_points`` is located."""
        return self._cpp_object.dest_cells


class BoundingBoxTree:
    """Bounding box trees used in collision detection."""

    _cpp_object: typing.Union[
        _cpp.geometry.BoundingBoxTree_float32, _cpp.geometry.BoundingBoxTree_float64
    ]

    def __init__(self, tree):
        """Wrap a C++ BoundingBoxTree.

        Note:
            This initializer should not be used in user code. Use
                ``bb_tree``.

        """
        self._cpp_object = tree

    @property
    def num_bboxes(self) -> int:
        """Number of bounding boxes."""
        return self._cpp_object.num_bboxes

    def get_bbox(self, i) -> npt.NDArray[np.floating]:
        """Get lower and upper corners of the ith bounding box.

        Args:
            i: Index of the box.

        Returns:
            The 'lower' and 'upper' points of the bounding box.
            Shape is ``(2, 3)``,

        """
        return self._cpp_object.get_bbox(i)

    def create_global_tree(self, comm) -> BoundingBoxTree:
        return BoundingBoxTree(self._cpp_object.create_global_tree(comm))


def bb_tree(
    mesh: Mesh,
    dim: int,
    entities: typing.Optional[npt.NDArray[np.int32]] = None,
    padding: float = 0.0,
) -> BoundingBoxTree:
    """Create a bounding box tree for use in collision detection.

    Args:
        mesh: The mesh.
        dim: Dimension of the mesh entities to build bounding box for.
        entities: List of entity indices (local to process). If not
            supplied, all owned and ghosted entities are used.
        padding: Padding for each bounding box.

    Returns:
        Bounding box tree.

    """
    map = mesh.topology.index_map(dim)
    if map is None:
        raise RuntimeError(f"Mesh entities of dimension {dim} have not been created.")
    if entities is None:
        entities = np.arange(map.size_local + map.num_ghosts, dtype=np.int32)

    dtype = mesh.geometry.x.dtype
    if np.issubdtype(dtype, np.float32):
        return BoundingBoxTree(
            _cpp.geometry.BoundingBoxTree_float32(mesh._cpp_object, dim, entities, padding)
        )
    elif np.issubdtype(dtype, np.float64):
        return BoundingBoxTree(
            _cpp.geometry.BoundingBoxTree_float64(mesh._cpp_object, dim, entities, padding)
        )
    else:
        raise NotImplementedError(f"Type {dtype} not supported.")


def compute_collisions_trees(
    tree0: BoundingBoxTree, tree1: BoundingBoxTree
) -> npt.NDArray[np.int32]:
    """Compute all collisions between two bounding box trees.

    Args:
        tree0: First bounding box tree.
        tree1: Second bounding box tree.

    Returns:
        List of pairs of intersecting box indices from each tree. Shape
        is ``(num_collisions, 2)``.

    """
    return _cpp.geometry.compute_collisions_trees(tree0._cpp_object, tree1._cpp_object)


def compute_collisions_points(tree: BoundingBoxTree, x: npt.NDArray[np.floating]) -> AdjacencyList:
    """Compute collisions between points and leaf bounding boxes.

    Bounding boxes can overlap, therefore points can collide with more
    than one box.

    Args:
        tree: Bounding box tree.
        x: Points (``shape=(num_points, 3)``).

    Returns:
       For each point, the bounding box leaves that collide with the
       point.

    """
    return AdjacencyList(_cpp.geometry.compute_collisions_points(tree._cpp_object, x))


def compute_closest_entity(
    tree: BoundingBoxTree,
    midpoint_tree: BoundingBoxTree,
    mesh: Mesh,
    points: npt.NDArray[np.floating],
) -> npt.NDArray[np.int32]:
    """Compute closest mesh entity to a point.

    Args:
        tree: bounding box tree for the entities.
        midpoint_tree: A bounding box tree with the midpoints of all
            the mesh entities. This is used to accelerate the search.
        mesh: The mesh.
        points: The points to check for collision, ``shape=(num_points,3)``.

    Returns:
        Mesh entity index for each point in ``points``. Returns -1 for a
        point if the bounding box tree is empty.

    """
    return _cpp.geometry.compute_closest_entity(
        tree._cpp_object, midpoint_tree._cpp_object, mesh._cpp_object, points
    )


def create_midpoint_tree(mesh: Mesh, dim: int, entities: npt.NDArray[np.int32]) -> BoundingBoxTree:
    """Create a bounding box tree for the midpoints of a subset of entities.

    Args:
        mesh: The mesh.
        dim: Topological dimension of the entities.
        entities: Indices of mesh entities to include.

    Returns:
        Bounding box tree for midpoints of cell entities.

    """
    return BoundingBoxTree(_cpp.geometry.create_midpoint_tree(mesh._cpp_object, dim, entities))


def compute_colliding_cells(
    mesh: Mesh, candidates: AdjacencyList, x: npt.NDArray[np.floating]
) -> AdjacencyList:
    """From a mesh, find which cells collide with a set of points.

    Args:
        mesh: The mesh.
        candidate_cells: Adjacency list of candidate colliding cells for
            the ith point in ``x``.
        points: The points to check for collision ``shape=(num_points, 3)``,

    Returns:
        Adjacency list where the ith node is the list of entities that
        collide with the ith point.

    """
    return AdjacencyList(
        _cpp.geometry.compute_colliding_cells(mesh._cpp_object, candidates._cpp_object, x)
    )


def squared_distance(
    mesh: Mesh, dim: int, entities: npt.NDArray[np.int32], points: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Compute the squared distance between a point and a mesh entity.

    The distance is computed between the ith input points and the ith
    input entity.

    Args:
        mesh: Mesh containing the entities.
        dim: Topological dimension of the mesh entities.
        entities: Indices of the mesh entities (local to process).
        points: Points to compute the shortest distance from
            (``shape=(num_points, 3)``).

    Returns:
        Squared shortest distance from ``points[i]`` to ``entities[i]``.

    """
    return _cpp.geometry.squared_distance(mesh._cpp_object, dim, entities, points)


def compute_distance_gjk(
    p: npt.NDArray[np.floating], q: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Compute the distance between two convex bodies p and q, each defined by a set of points.

    Uses the Gilbert-Johnson-Keerthi (GJK) distance algorithm.

    Args:
        p: Body 1 list of points (``shape=(num_points, gdim)``).
        q: Body 2 list of points (``shape=(num_points, gdim)``).

    Returns:
        Shortest vector between the two bodies.

    """
    return _cpp.geometry.compute_distance_gjk(p, q)
