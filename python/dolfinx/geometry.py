# Copyright (C) 2018-2021 Michal Habera, Garth N. Wells and
# Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Methods for geometric searches and operations."""

from __future__ import annotations

import typing

from mpi4py import MPI as _MPI

import numpy as np
import numpy.typing as npt

if typing.TYPE_CHECKING:
    from dolfinx.mesh import Mesh

from dolfinx import cpp as _cpp
from dolfinx.graph import AdjacencyList
from dolfinx.typing import Real

__all__ = [
    "BoundingBoxTree",
    "PointOwnershipData",
    "bb_tree",
    "compute_closest_entity",
    "compute_colliding_cells",
    "compute_collisions_points",
    "compute_collisions_trees",
    "compute_distance_gjk",
    "compute_distances_gjk",
    "create_midpoint_tree",
    "determine_point_ownership",
    "squared_distance",
]


class PointOwnershipData(typing.Generic[Real]):
    """Class for storing data related to the ownership of points."""

    _cpp_object: _cpp.geometry.PointOwnershipData_float32 | _cpp.geometry.PointOwnershipData_float64

    def __init__(
        self,
        ownership_data: _cpp.geometry.PointOwnershipData_float32
        | _cpp.geometry.PointOwnershipData_float64,
    ) -> None:
        """Wrap a C++ PointOwnershipData."""
        self._cpp_object = ownership_data

    @property
    def src_owner(self) -> npt.NDArray[np.int32]:
        """Ranks owning points sent into ownership determination."""
        return self._cpp_object.src_owner

    @property
    def dest_owner(self) -> npt.NDArray[np.int32]:
        """Ranks that sent ``dest_points`` to current process."""
        return self._cpp_object.dest_owners

    @property
    def dest_points(self) -> npt.NDArray[Real]:
        """Points owned by current rank."""
        return self._cpp_object.dest_points  # type: ignore[return-value]

    @property
    def dest_cells(self) -> npt.NDArray[np.int32]:
        """Cell indices where each entry of ``dest_points`` is located."""
        return self._cpp_object.dest_cells


class BoundingBoxTree(typing.Generic[Real]):
    """Bounding box trees used in collision detection."""

    _cpp_object: _cpp.geometry.BoundingBoxTree_float32 | _cpp.geometry.BoundingBoxTree_float64

    def __init__(
        self, tree: _cpp.geometry.BoundingBoxTree_float32 | _cpp.geometry.BoundingBoxTree_float64
    ) -> None:
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

    @property
    def bbox_coordinates(self) -> npt.NDArray[Real]:
        """Coordinates of lower and upper corners of bounding boxes.

        Note:
            Rows `2*ibbox` and `2*ibbox+1` correspond to the lower
            and upper corners of bounding box `ibbox`, respectively.
        """
        return self._cpp_object.bbox_coordinates  # type: ignore[return-value]

    def get_bbox(self, i: int) -> npt.NDArray[Real]:
        """Get lower and upper corners of the ith bounding box.

        Args:
            i: Index of the box.

        Returns:
            The 'lower' and 'upper' points of the bounding box.
            Shape is ``(2, 3)``,

        """
        return self._cpp_object.get_bbox(i)  # type: ignore[return-value]

    def create_global_tree(self, comm: _MPI.Comm) -> BoundingBoxTree[Real]:
        """Create a global bounding box tree."""
        return BoundingBoxTree(self._cpp_object.create_global_tree(comm))


def bb_tree(
    mesh: Mesh[Real],
    dim: int,
    *,
    padding: float = 0.0,
    entities: npt.NDArray[np.int32] | None = None,
) -> BoundingBoxTree[Real]:
    """Create a bounding box tree for use in collision detection.

    Args:
        mesh: The mesh.
        dim: Dimension of the mesh entities to build bounding box for.
        padding: Padding for each bounding box.
        entities: List of entity indices (local to process). If not
            supplied, all owned and ghosted entities are used.

    Returns:
        Bounding box tree.

    """
    map = mesh.topology.index_map(dim)
    if map is None:
        raise RuntimeError(f"Mesh entities of dimension {dim} have not been created.")

    cpp_mesh = mesh._cpp_object
    if isinstance(cpp_mesh, _cpp.mesh.Mesh_float32):
        return BoundingBoxTree(
            _cpp.geometry.BoundingBoxTree_float32(cpp_mesh, dim, padding, entities)
        )
    elif isinstance(cpp_mesh, _cpp.mesh.Mesh_float64):
        return BoundingBoxTree(
            _cpp.geometry.BoundingBoxTree_float64(cpp_mesh, dim, padding, entities)
        )
    else:
        raise NotImplementedError(f"Type {mesh.geometry.x.dtype} not supported.")


def compute_collisions_trees(
    tree0: BoundingBoxTree[Real], tree1: BoundingBoxTree[Real]
) -> npt.NDArray[np.int32]:
    """Compute all collisions between two bounding box trees.

    Args:
        tree0: First bounding box tree.
        tree1: Second bounding box tree.

    Returns:
        List of pairs of intersecting box indices from each tree. Shape
        is ``(num_collisions, 2)``.

    """
    return _cpp.geometry.compute_collisions_trees(tree0._cpp_object, tree1._cpp_object)  # type: ignore[arg-type]


def compute_collisions_points(tree: BoundingBoxTree[Real], x: npt.NDArray[Real]) -> AdjacencyList:
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
    return AdjacencyList(_cpp.geometry.compute_collisions_points(tree._cpp_object, x))  # type: ignore[arg-type]


def compute_closest_entity(
    tree: BoundingBoxTree[Real],
    midpoint_tree: BoundingBoxTree[Real],
    mesh: Mesh[Real],
    points: npt.NDArray[Real],
) -> npt.NDArray[np.int32]:
    """Compute closest mesh entity to a point.

    Args:
        tree: bounding box tree for the entities.
        midpoint_tree: A bounding box tree with the midpoints of all
            the mesh entities. This is used to accelerate the search.
        mesh: The mesh.
        points: The points to check for collision,
            ``shape=(num_points,3)``.

    Returns:
        Mesh entity index for each point in ``points``. Returns -1 for a
        point if the bounding box tree is empty.

    """
    return _cpp.geometry.compute_closest_entity(
        tree._cpp_object,  # type: ignore[arg-type]
        midpoint_tree._cpp_object,  # type: ignore[arg-type]
        mesh._cpp_object,  # type: ignore[arg-type]
        points,  # type: ignore[arg-type]
    )


def create_midpoint_tree(
    mesh: Mesh[Real], dim: int, entities: npt.NDArray[np.int32]
) -> BoundingBoxTree[Real]:
    """Create bounding box tree for the midpoints of a subset of entities.

    Args:
        mesh: The mesh.
        dim: Topological dimension of the entities.
        entities: Indices of mesh entities to include.

    Returns:
        Bounding box tree for midpoints of cell entities.
    """
    return BoundingBoxTree(_cpp.geometry.create_midpoint_tree(mesh._cpp_object, dim, entities))


def compute_colliding_cells(
    msh: Mesh[Real], candidates: AdjacencyList, x: npt.NDArray[Real]
) -> AdjacencyList:
    """From a mesh, find which cells collide with a set of points.

    Args:
        msh: The mesh.
        candidates: Adjacency list of candidate colliding cells for
            the ith point in ``x``.
        x: Points to check for collision ``shape=(num_points, 3)``,

    Returns:
        Adjacency list where the ith node is the list of entities that
        collide with the ith point.

    """
    return AdjacencyList(
        _cpp.geometry.compute_colliding_cells(msh._cpp_object, candidates._cpp_object, x)  # type: ignore[arg-type]
    )


def squared_distance(
    mesh: Mesh[Real], dim: int, entities: npt.NDArray[np.int32], points: npt.NDArray[Real]
) -> npt.NDArray[Real]:
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
    return _cpp.geometry.squared_distance(mesh._cpp_object, dim, entities, points)  # type: ignore[arg-type,return-value]


def compute_distance_gjk(p: npt.NDArray[Real], q: npt.NDArray[Real]) -> npt.NDArray[Real]:
    """Compute the distance between two convex bodies.

    Each body is defined by a set of points. Uses the
    Gilbert-Johnson-Keerthi (GJK) distance algorithm.

    Args:
        p: Body 1 list of points (``shape=(num_points, gdim)``).
        q: Body 2 list of points (``shape=(num_points, gdim)``).

    Returns:
        Shortest vector between the two bodies.
    """
    if p.dtype != q.dtype:
        raise ValueError("p and q must have the same dtype.")
    if np.issubdtype(p.dtype, np.float32):
        return _cpp.geometry.compute_distance_gjk_float32(p, q)  # type: ignore[arg-type,return-value]
    elif np.issubdtype(p.dtype, np.float64):
        return _cpp.geometry.compute_distance_gjk_float64(p, q)  # type: ignore[arg-type,return-value]
    raise RuntimeError("Invalid dtype in compute_distance_gjk")


def compute_distances_gjk(
    bodies: list[npt.NDArray[Real]], q: npt.NDArray[Real], num_threads: int
) -> npt.NDArray[Real]:
    """Compute the distance between a set of convex bodies.

    For each convex body defined in `bodies`;
    (a set of 3D points for each body) find the shortest distance vector
    to the body `q` defined by another set of 3D points.
    The method uses the
    Gilbert-Johnson-Keerthi (GJK) distance algorithm.

    Args:
        bodies: List of bodies, where each body is an array of
            (``shape=(num_points_i, 3, gdim)``).
        q: Body 2 list of points (``shape=(num_points_2, 3)``).
        num_threads: Number of threads to use for GJK computation.

    Returns:
        Shortest vector between the two bodies.
    """
    if not all(p.dtype == q.dtype for p in bodies):
        raise ValueError("All bodies and q must have the same dtype.")
    if np.issubdtype(q.dtype, np.float32):
        return _cpp.geometry.compute_distances_gjk_float32(bodies, q, num_threads)  # type: ignore[arg-type,return-value]
    elif np.issubdtype(q.dtype, np.float64):
        return _cpp.geometry.compute_distances_gjk_float64(bodies, q, num_threads)  # type: ignore[arg-type,return-value]
    raise RuntimeError("Invalid dtype in compute_distances_gjk")


def determine_point_ownership(
    mesh: Mesh,
    points: npt.NDArray[Real],
    padding: float,
    cells: npt.NDArray[np.int32] | None = None,
    find_closest_cell: bool = True,
) -> PointOwnershipData[Real]:
    """Determine, for each point, the owning process of a containing cell.

    A cell is a *candidate* for a point if the cell's bounding box,
    padded by ``padding``, contains the point. Each candidate is then
    tested for actual containment of the point with the GJK algorithm.
    If no candidate actually contains a point, the point is either left
    unowned or, if ``find_closest_cell`` is ``True``, assigned to the
    candidate cell closest to it (by GJK distance).

    Args:
        mesh: The mesh
        points: Points to check for collision, ``shape=(num_points, gdim)``
        padding: Amount of absolute padding applied to each cell's
            bounding box before searching for candidate cells/processes.
            Increasing ``padding`` increases the number of cells
            considered as candidates for a point; it does not by
            itself decide whether a point with no actually-containing
            cell is assigned an owner, which is controlled by
            ``find_closest_cell``.
        cells: Cells to check for ownership
            If ``None`` then all cells are considered.
        find_closest_cell: If ``True`` (default), a point not
            actually contained in any candidate cell is instead
            assigned to the process owning the candidate cell closest
            to it. If ``False``, such a point is left unowned.

    Returns:
        Point ownership data

    Note:
        ``dest_owner`` is sorted

        An entry of ``src_owner`` is ``-1`` if the corresponding point
        was not contained in any candidate cell and, if
        ``find_closest_cell`` is ``True``, had no candidate cell to
        fall back on either (e.g. because ``padding`` was too small).

        With ``find_closest_cell`` set to ``True``, a large padding
            value will increase the run-time of the code by orders of
            magnitude. General advice is to use a padding on the scale
            of the cell size.
    """
    return PointOwnershipData(
        _cpp.geometry.determine_point_ownership(
            mesh._cpp_object,  # type: ignore[arg-type]
            points,  # type: ignore[arg-type]
            padding,
            cells,
            find_closest_cell,
        )
    )
