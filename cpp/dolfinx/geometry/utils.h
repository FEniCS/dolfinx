// Copyright (C) 2019-2021 Garth N. Wells and Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cstdint>
#include <dolfinx/graph/AdjacencyList.h>
#include <span>
#include <vector>

namespace dolfinx::mesh
{
class Mesh;
}

namespace dolfinx::geometry
{
class BoundingBoxTree;

/// Create a bounding box tree for a subset of entities (local to
/// process) based on the entity midpoints
/// @param[in] mesh The mesh
/// @param[in] tdim The topological dimension of the entity
/// @param[in] entity_indices List of local entity indices
/// @return Bounding box tree for midpoints of mesh entities
BoundingBoxTree
create_midpoint_tree(const mesh::Mesh& mesh, int tdim,
                     const std::span<const std::int32_t>& entity_indices);

/// Compute all collisions between two BoundingBoxTrees (local to
/// process)
/// @param[in] tree0 First BoundingBoxTree
/// @param[in] tree1 Second BoundingBoxTree
/// @return List of pairs of intersecting box indices from each tree, flattened
/// as a vector of size num_intersections*2
std::vector<std::int32_t> compute_collisions(const BoundingBoxTree& tree0,
                                             const BoundingBoxTree& tree1);

/// Compute all collisions between bounding boxes and for a set of
/// points
/// @param[in] tree The bounding box tree
/// @param[in] points The points (shape=(num_points, 3)). Storage
/// is row-major.
/// @return An adjacency list where the ith node corresponds to the
/// bounding box leaves that contain the ith point
graph::AdjacencyList<std::int32_t>
compute_collisions(const BoundingBoxTree& tree,
                   const std::span<const double>& points);

/// Compute the first collision between a point and
/// the cells of the mesh
/// @param[in] mesh The mesh
/// @param[in] tree The bounding box tree
/// @param[in] point The point (shape=(3))
/// @return The local cell index, -1 if not found
int compute_first_colliding_cell(const mesh::Mesh& mesh,
                                 const BoundingBoxTree& tree,
                                 const std::array<double, 3>& point);

/// Compute closest mesh entity to a point
/// @param[in] tree The bounding box tree for the entities
/// @param[in] midpoint_tree A bounding box tree with the midpoints of
/// all the mesh entities. This is used to accelerate the search
/// @param[in] mesh The mesh
/// @param[in] points The set of points (shape=(num_points, 3)). Storage
/// is row-major.
/// @return Index of the closest mesh entity to a point. The ith entry
/// is the index of the closest entity to the ith input point.
/// @note Returns index -1 if the bounding box tree is empty
std::vector<std::int32_t> compute_closest_entity(
    const BoundingBoxTree& tree, const BoundingBoxTree& midpoint_tree,
    const mesh::Mesh& mesh, const std::span<const double>& points);

/// Compute squared distance between point and bounding box
/// @param[in] b Bounding box coordinates
/// @param[in] x A point
/// @return The shortest distance between the bounding box `b` and the
/// point `x`. Returns zero if `x` is inside box.
double compute_squared_distance_bbox(std::span<const double, 6> b,
                                     std::span<const double, 3> x);

/// Compute the shortest vector from a mesh entity to a point
/// @param[in] mesh The mesh
/// @param[in] dim The topological dimension of the mesh entity
/// @param[in] entities The list of entities (local to process)
/// @param[in] points The set of points (shape=(num_points, 3)), using row-major
/// storage
/// @return An array of vectors (shape=(num_points, 3)) where the ith
/// row is the shortest vector between the ith entity and the ith point.
/// Storage is row-major.
std::vector<double>
shortest_vector(const mesh::Mesh& mesh, int dim,
                const std::span<const std::int32_t>& entities,
                const std::span<const double>& points);

/// Compute the squared distance between a point and a mesh entity. The
/// distance is computed between the ith input points and the ith input
/// entity.
/// @note Uses the GJK algorithm, see geometry::compute_distance_gjk for
/// details.
/// @note Uses a convex hull approximation of linearized geometry
/// @param[in] mesh Mesh containing the entities
/// @param[in] dim The topological dimension of the mesh entities
/// @param[in] entities The indices of the mesh entities (local to process)
/// @param[in] points The set points from which to computed the shortest
/// (shape=(num_points, 3)). Storage is row-major.
/// @return Squared shortest distance from points[i] to entities[i]
std::vector<double>
squared_distance(const mesh::Mesh& mesh, int dim,
                 const std::span<const std::int32_t>& entities,
                 const std::span<const double>& points);

/// From a Mesh, find which cells collide with a set of points.
/// @note Uses the GJK algorithm, see geometry::compute_distance_gjk for
/// details
/// @param[in] mesh The mesh
/// @param[in] candidate_cells List of candidate colliding cells for the
/// ith point in `points`
/// @param[in] points The points to check for collision
/// (shape=(num_points, 3)). Storage is row-major.
/// @return Adjacency list where the ith node is the list of entities
/// that collide with the ith point
/// @note There may be nodes with no entries in the adjacency list
graph::AdjacencyList<std::int32_t> compute_colliding_cells(
    const mesh::Mesh& mesh,
    const graph::AdjacencyList<std::int32_t>& candidate_cells,
    const std::span<const double>& points);

/// Given a set of points (local on each process) which process is colliding,
/// using the GJK algorithm on cells to determine collisions.
/// @param[in] mesh The mesh
/// @param[in] points The points to check for collision (shape=(num_points, 3)).
/// Storage is row-major.
/// @return Quadratuplet (src_owner, dest_owner, dest_points, dest_cells), where
/// src_owner is a list of ranks corresponding to the input points. dest_owner
/// is a list of ranks corresponding to dest_points, the points that this
/// process owns. dest_cells contains the corresponding cell for each entry in
/// dest_points.
/// @note dest_owner is sorted
/// @note Returns -1 if no colliding process is found
/// @note dest_points is flattened row-major, shape (dest_owner.size(), 3)
/// @note Only looks through cells owned by the process
std::tuple<std::vector<std::int32_t>, std::vector<std::int32_t>,
           std::vector<double>, std::vector<std::int32_t>>
determine_point_ownership(const mesh::Mesh& mesh,
                          std::span<const double> points);

} // namespace dolfinx::geometry
