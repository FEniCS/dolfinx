// Copyright (C) 2019-2021 Garth N. Wells and Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <utility>
#include <vector>
#include <xtl/xspan.hpp>

namespace dolfinx
{
namespace mesh
{
class Mesh;
} // namespace mesh

namespace geometry
{
class BoundingBoxTree;

/// Create a bounding box tree for a subset of entities (local to process) based
/// on the entity midpoints
/// @param[in] mesh The mesh
/// @param[in] tdim The topological dimension of the entity
/// @param[in] entity_indices List of local entity indices
/// @return Bounding box tree for midpoints of mesh entities
BoundingBoxTree
create_midpoint_tree(const mesh::Mesh& mesh, int tdim,
                     const std::vector<std::int32_t>& entity_indices);

/// Compute all collisions between two BoundingBoxTrees (local to process).
/// @param[in] tree0 First BoundingBoxTree
/// @param[in] tree1 Second BoundingBoxTree
/// @return List of pairs of intersecting box indices (local to process) from
/// each tree
std::vector<std::array<int, 2>>
compute_collisions(const BoundingBoxTree& tree0, const BoundingBoxTree& tree1);

/// Compute all collisions between bounding boxes and point
/// @param[in] tree The bounding box tree
/// @param[in] p The point
/// @return Bounding box leaves (local to process) that contain the point
std::vector<int> compute_collisions(const BoundingBoxTree& tree,
                                    const std::array<double, 3>& p);

/// Compute closest mesh entity (local to process) for the topological distance
/// of the bounding box tree and distance and a point
/// @param[in] tree The bounding box tree
/// @param[in] p The point
/// @param[in] mesh The mesh
/// @param[in] R Radius for search. Supplying a negative radius causes
/// the function to estimate an initial search radius.
/// @return The local index of the entity and the distance from the point.
std::pair<std::int32_t, double>
compute_closest_entity(const BoundingBoxTree& tree,
                       const std::array<double, 3>& p, const mesh::Mesh& mesh,
                       double R = -1);

/// Compute squared distance between point and bounding box wih index
/// "node". Returns zero if point is inside box.
double
compute_squared_distance_bbox(const std::array<std::array<double, 3>, 2>& b,
                              const std::array<double, 3>& x);

/// Compute squared distance from a given point to the nearest point on
/// a cell (only first order convex cells are supported at this stage)
/// Uses the GJK algorithm, see geometry::compute_distance_gjk for
/// details.
///
/// @note Currently a convex hull approximation of linearized geometry.
///
/// @param[in] mesh Mesh containing the mesh entity
/// @param[in] dim The topological dimension of the mesh entity
/// @param[in] index The index of the mesh entity (local to process)
/// @param[in] p The point from which to compouted the shortest distance
/// to the mesh to compute the Point
/// @return shortest squared distance from p to entity
double squared_distance(const mesh::Mesh& mesh, int dim, std::int32_t index,
                        const std::array<double, 3>& p);

/// From the given Mesh, select up to n cells (local to process) from the list
/// which actually collide with point p. n may be zero (selects all valid
/// cells). Less than n cells may be returned.
/// @param[in] mesh Mesh
/// @param[in] candidate_cells List of cell indices to test
/// @param[in] p Point to check for collision
/// @param[in] n Maximum number of positive results to return
/// @return List of cells which collide with point
std::vector<std::int32_t>
select_colliding_cells(const dolfinx::mesh::Mesh& mesh,
                       const xtl::span<const std::int32_t>& candidate_cells,
                       const std::array<double, 3>& p, int n);
} // namespace geometry
} // namespace dolfinx
