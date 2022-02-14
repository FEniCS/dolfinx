# Copyright (C) 2018-2021 Michal Habera, Garth N. Wells and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Methods for geometric searches and operations"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from dolfinx.mesh import Mesh
    from dolfinx.cpp.graph import AdjacencyList_int32

import numpy

from dolfinx import cpp as _cpp
from dolfinx.cpp.geometry import (compute_closest_entity, compute_collisions,
                                  compute_distance_gjk, create_midpoint_tree)

__all__ = ["compute_colliding_cells", "squared_distance", "compute_closest_entity", "compute_collisions",
           "compute_distance_gjk", "create_midpoint_tree"]


class BoundingBoxTree(_cpp.geometry.BoundingBoxTree):
    """A class for representing bounding box trees used in collision
    detection."""

    def __init__(self, mesh: Mesh, dim: int, entities=None, padding: float = 0.0):
        """Create a bounding box tree for entities of a mesh.

        Args:
            mesh: The mesh
            dim: The dimension of the mesh entities
            entities: List of entity indices (local to process). If not supplied,
                all owned and ghosted entities are used.
            padding: Padding for each bounding box

        """
        map = mesh.topology.index_map(dim)
        if map is None:
            raise RuntimeError(f"Mesh entities of dimension {dim} have not been created.")
        if entities is None:
            entities = range(0, map.size_local + map.num_ghosts)

        super().__init__(mesh, dim, entities, padding)


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
    return _cpp.geometry.compute_colliding_cells(mesh, candidates, x)


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
    return _cpp.geometry.squared_distance(mesh, dim, entities, points)
