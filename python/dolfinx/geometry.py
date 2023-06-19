# Copyright (C) 2018-2021 Michal Habera, Garth N. Wells and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Methods for geometric searches and operations"""

from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt

if typing.TYPE_CHECKING:
    from dolfinx.mesh import Mesh
    from dolfinx.cpp.graph import AdjacencyList_int32

import numpy

from dolfinx import cpp as _cpp
from dolfinx.cpp.geometry import compute_distance_gjk

__all__ = ["compute_colliding_cells", "squared_distance", "compute_closest_entity", "compute_collisions",
           "compute_distance_gjk", "create_midpoint_tree"]


class BoundingBoxTree:
    """Bounding box trees used in collision detection."""

    def __init__(self, mesh: Mesh, dim: int, entities=None, padding: float = 0.0):
        """Create a bounding box tree for entities of a mesh.

        Args:
            mesh: The mesh
            dim: The dimension of the mesh entities
            entities: List of entity indices (local to process). If not supplied,
                all owned and ghosted entities are used.
            padding: Padding for each bounding box

        """
        if mesh.geometry.x.dtype == np.float32:
            _bbtree = _cpp.geometry.BoundingBoxTree_float32
        elif mesh.geometry.x.dtype == np.float64:
            _bbtree = _cpp.geometry.BoundingBoxTree_float64

        map = mesh.topology.index_map(dim)
        if map is None:
            raise RuntimeError(f"Mesh entities of dimension {dim} have not been created.")
        if entities is None:
            entities = range(0, map.size_local + map.num_ghosts)

        self._cpp_object = _bbtree(mesh._cpp_object, dim, entities, padding)

    @property
    def num_bboxes(self):
        return self._cpp_object.num_bboxes

    def get_bbox(self, i):
        return self._cpp_object.get_bbox(i)

    def create_global_tree(self, comm):
        return self._cpp_object.create_global_tree(comm)


def compute_collisions(bb0, bb1):
    try:
        return _cpp.geometry.compute_collisions(bb0._cpp_object, bb1._cpp_object)
    except AttributeError:
        return _cpp.geometry.compute_collisions(bb0._cpp_object, bb1)


def compute_closest_entity(tree: BoundingBoxTree, midpoint_tree: BoundingBoxTree, mesh: Mesh,
                           points: numpy.ndarray) -> npt.NDArray[np.int32]:
    """Compute closest mesh entity to a point.

        Args:
            tree: bounding box tree for the entities
            midpoint_tree: A bounding box tree with the midpoints of all
                the mesh entities. This is used to accelerate the search.
            mesh: The mesh
            points: The points to check for collision, shape=(num_points, 3)

        Returns:
            Mesh entity index for each point in `points`. Returns -1 for
            a point if the bounding box tree is empty.

    """
    return _cpp.geometry.compute_closest_entity(tree, midpoint_tree, mesh._cpp_object, points)


def create_midpoint_tree(mesh: Mesh, dim: int, entities: numpy.ndarray):
    return _cpp.geometry.create_midpoint_tree(mesh._cpp_object, dim, entities)


def compute_colliding_cells(mesh: Mesh, candidates: AdjacencyList_int32, x: numpy.ndarray):
    """From a mesh, find which cells collide with a set of points.

    Args:
        mesh: The mesh
        candidate_cells: Adjacency list of candidate colliding cells for
            the ith point in `x`
        points: The points to check for collision shape=(num_points, 3)

    Returns:
        Adjacency list where the ith node is the list of entities that
        collide with the ith point
    """
    return _cpp.geometry.compute_colliding_cells(mesh._cpp_object, candidates, x)


def squared_distance(mesh: Mesh, dim: int, entities: typing.List[int], points: numpy.ndarray):
    """Compute the squared distance between a point and a mesh entity.

    The distance is computed between the ith input points and the ith
    input entity.

    Args:
        mesh: Mesh containing the entities
        dim: The topological dimension of the mesh entities
        entities: Indices of the mesh entities (local to process)
        points: Set points from which to computed the shortest distance
            (``shape=(num_points, 3)``)

    Returns:
        Squared shortest distance from points[i] to entities[i]

    """
    return _cpp.geometry.squared_distance(mesh._cpp_object, dim, entities, points)
