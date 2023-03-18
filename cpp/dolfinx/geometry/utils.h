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
template <typename T>
class Mesh;
}

namespace dolfinx::geometry
{
class BoundingBoxTree;

/// @brief Create a bounding box tree for the midpoints of a subset of
/// entities.
/// @param[in] mesh The mesh
/// @param[in] tdim The topological dimension of the entity
/// @param[in] entity_indices List of local entity indices
/// @return Bounding box tree for midpoints of entities
BoundingBoxTree
create_midpoint_tree(const mesh::Mesh<double>& mesh, int tdim,
                     std::span<const std::int32_t> entity_indices);

/// @brief Compute all collisions between two bounding box trees.
/// @param[in] tree0 First BoundingBoxTree
/// @param[in] tree1 Second BoundingBoxTree
/// @return List of pairs of intersecting box indices from each tree,
/// flattened as a vector of size num_intersections*2
std::vector<std::int32_t> compute_collisions(const BoundingBoxTree& tree0,
                                             const BoundingBoxTree& tree1);

/// @brief Compute collisions between points and leaf bounding boxes.
///
/// Bounding boxes can overlap, therefore points can collide with more
/// than one box.
///
/// @param[in] tree The bounding box tree
/// @param[in] points The points (`shape=(num_points, 3)`). Storage is
/// row-major.
/// @return For each point, the bounding box leaves that collide with
/// the point.
graph::AdjacencyList<std::int32_t>
compute_collisions(const BoundingBoxTree& tree, std::span<const double> points);

/// @brief Compute the cell that collides with a point.
///
/// A point can collide with more than one cell. The first cell detected
/// to collide with the point is returned. If no collision is detected,
/// -1 is returned.
///
/// @param[in] mesh The mesh
/// @param[in] tree The bounding box tree
/// @param[in] point The point (`shape=(3,)`)
/// @return The local cell index, -1 if not found
int compute_first_colliding_cell(const mesh::Mesh<double>& mesh,
                                 const BoundingBoxTree& tree,
                                 const std::array<double, 3>& point);

/// @brief Compute closest mesh entity to a point.
///
/// @note Returns a vector filled with index -1 if the bounding box tree
/// is empty.
///
/// @param[in] tree The bounding box tree for the entities
/// @param[in] midpoint_tree A bounding box tree with the midpoints of
/// all the mesh entities. This is used to accelerate the search.
/// @param[in] mesh The mesh
/// @param[in] points The set of points (`shape=(num_points, 3)`).
/// Storage is row-major.
/// @return For each point, the index of the closest mesh entity.
std::vector<std::int32_t> compute_closest_entity(
    const BoundingBoxTree& tree, const BoundingBoxTree& midpoint_tree,
    const mesh::Mesh<double>& mesh, std::span<const double> points);

/// @brief Compute squared distance between point and bounding box.
///
/// @param[in] b Bounding box coordinates
/// @param[in] x A point
/// @return The shortest distance between the bounding box `b` and the
/// point `x`. Returns zero if `x` is inside box.
double compute_squared_distance_bbox(std::span<const double, 6> b,
                                     std::span<const double, 3> x);

/// @brief Compute the shortest vector from a mesh entity to a point.
///
/// @param[in] mesh The mesh
/// @param[in] dim Topological dimension of the mesh entity
/// @param[in] entities List of entities
/// @param[in] points Set of points (`shape=(num_points, 3)`), using
/// row-major storage.
/// @return An array of vectors (shape=(num_points, 3)) where the ith
/// row is the shortest vector between the ith entity and the ith point.
/// Storage is row-major.
std::vector<double> shortest_vector(const mesh::Mesh<double>& mesh, int dim,
                                    std::span<const std::int32_t> entities,
                                    std::span<const double> points);

/// @brief Compute the squared distance between a point and a mesh
/// entity.
///
/// The distance is computed between the ith input points and the ith
/// input entity.
///
/// @note Uses the GJK algorithm, see geometry::compute_distance_gjk for
/// details.
/// @note Uses a convex hull approximation of linearized geometry
/// @param[in] mesh Mesh containing the entities
/// @param[in] dim The topological dimension of the mesh entities
/// @param[in] entities The indices of the mesh entities (local to process)
/// @param[in] points The set points from which to computed the shortest
/// (shape=(num_points, 3)). Storage is row-major.
/// @return Squared shortest distance from points[i] to entities[i]
std::vector<double> squared_distance(const mesh::Mesh<double>& mesh, int dim,
                                     std::span<const std::int32_t> entities,
                                     std::span<const double> points);

/// @brief Compute which cells collide with a point.
///
/// @note Uses the GJK algorithm, see geometry::compute_distance_gjk for
/// details.
///
/// @param[in] mesh The mesh
/// @param[in] candidate_cells List of candidate colliding cells for the
/// ith point in `points`
/// @param[in] points Points to check for collision (`shape=(num_points,
/// 3)`). Storage is row-major.
/// @return For each point, the cells that collide with the point.
graph::AdjacencyList<std::int32_t> compute_colliding_cells(
    const mesh::Mesh<double>& mesh,
    const graph::AdjacencyList<std::int32_t>& candidate_cells,
    std::span<const double> points);

/// @brief Given a set of points, determine which process is colliding,
/// using the GJK algorithm on cells to determine collisions.
///
/// @todo This docstring is unclear. Needs fixing.
///
/// @param[in] mesh The mesh
/// @param[in] points Points to check for collision (`shape=(num_points,
/// 3)`). Storage is row-major.
/// @return Quadratuplet (src_owner, dest_owner, dest_points,
/// dest_cells), where src_owner is a list of ranks corresponding to the
/// input points. dest_owner is a list of ranks corresponding to
/// dest_points, the points that this process owns. dest_cells contains
/// the corresponding cell for each entry in dest_points.
///
/// @note dest_owner is sorted
/// @note Returns -1 if no colliding process is found
/// @note dest_points is flattened row-major, shape (dest_owner.size(), 3)
/// @note Only looks through cells owned by the process
std::tuple<std::vector<std::int32_t>, std::vector<std::int32_t>,
           std::vector<double>, std::vector<std::int32_t>>
determine_point_ownership(const mesh::Mesh<double>& mesh,
                          std::span<const double> points);

} // namespace dolfinx::geometry
