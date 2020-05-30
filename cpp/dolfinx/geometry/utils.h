// Copyright (C) 2019 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <utility>
#include <vector>

namespace dolfinx
{
namespace mesh
{
class Mesh;
} // namespace mesh

namespace geometry
{
class BoundingBoxTree;

/// Create a boundary box tree for cell midpoints
/// @param[in] mesh The mesh build tree of cell midpoints from
/// @return Bounding box tree for mesh cell midpoints
BoundingBoxTree create_midpoint_tree(const mesh::Mesh& mesh);

/// Compute all collisions between two BoundingBoxTrees.
/// @param[in] tree0 First BoundingBoxTree
/// @param[in] tree1 Second BoundingBoxTree
/// @return List of pairs of intersecting box indices from each tree
std::vector<std::array<int, 2>>
compute_collisions(const BoundingBoxTree& tree0, const BoundingBoxTree& tree1);

/// Compute all collisions between bounding boxes and point
/// @param[in] tree The bounding box tree
/// @param[in] p The point
/// @return Bounding box leaves that contain the point
std::vector<int> compute_collisions(const BoundingBoxTree& tree,
                                    const Eigen::Vector3d& p);

/// Compute all collisions between processes and Point returning a
/// list of process ranks
std::vector<int> compute_process_collisions(const BoundingBoxTree& tree,
                                            const Eigen::Vector3d& p);

/// Compute closest mesh entity and distance to the point. The tree must
/// have been initialised with topological co-dimension 0.
std::pair<int, double>
compute_closest_entity(const BoundingBoxTree& tree,
                       const BoundingBoxTree& tree_midpoint,
                       const Eigen::Vector3d& p, const mesh::Mesh& mesh);

/// Compute closest point and distance to a given point
/// @param[in] tree The bounding box tree. It must have been initialised
///   with topological dimension 0.
/// @param[in] p The point to compute the distance from
/// @return (point index, distance)
std::pair<int, double> compute_closest_point(const BoundingBoxTree& tree,
                                             const Eigen::Vector3d& p);

/// Compute squared distance between point and bounding box wih index
/// "node". Returns zero if point is inside box.
double compute_squared_distance_bbox(
    const Eigen::Array<double, 2, 3, Eigen::RowMajor>& b,
    const Eigen::Vector3d& x);

/// Compute squared distance from a given point to the nearest point on
/// a cell (only first order convex cells are supported at this stage)
/// Uses the GJK algorithm, see geometry::compute_distance_gjk for
/// details.
///
/// @note Currently a convex hull approximation of linearized geometry.
///
/// @param[in] mesh Mesh containing the mesh entity
/// @param[in] dim The topological dimension of the mesh entity
/// @param[in] index The index of the mesh entity
/// @param[in] p The point from which to compouted the shortest distance
///    to the mesh to compute the Point
/// @return shortest squared distance from p to entity
double squared_distance(const mesh::Mesh& mesh, int dim, std::int32_t index,
                        const Eigen::Vector3d& p);

/// From the given Mesh, select up to n cells from the list which actually
/// collide with point p. n may be zero (selects all valid cells). Less than n
/// cells may be returned.
/// @param[in] mesh Mesh
/// @param[in] candidate_cells List of cell indices to test
/// @param[in] point Point to check for collision
/// @param[in] n Maximum number of positive results to return
/// @return List of cells which collide with point
std::vector<int> select_colliding_cells(const dolfinx::mesh::Mesh& mesh,
                                        const std::vector<int>& candidate_cells,
                                        const Eigen::Vector3d& point, int n);
} // namespace geometry
} // namespace dolfinx
