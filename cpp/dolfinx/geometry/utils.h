// Copyright (C) 2019 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "BoundingBoxTree.h"
#include <Eigen/Dense>
#include <utility>
#include <vector>

namespace dolfinx
{
namespace mesh
{
class Mesh;
class MeshEntity;
} // namespace mesh

namespace geometry
{
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

/// Check whether bounding box a collides with bounding box (b)
bool bbox_in_bbox(const Eigen::Array<double, 2, 3, Eigen::RowMajor>& a,
                  const Eigen::Array<double, 2, 3, Eigen::RowMajor>& b,
                  double rtol = 1e-14);

/// Compute closest mesh entity and distance to the point. The tree must
/// have been initialised with topological co-dimension 0.
std::pair<int, double>
compute_closest_entity(const BoundingBoxTree& tree,
                       const BoundingBoxTree& tree_midpoint,
                       const Eigen::Vector3d& p, const mesh::Mesh& mesh);

/// Compute closest point and distance to a given point
/// @param[in] tree The bounding box tree. It must have been initialised
///                  with topological dimension 0.
/// @param[in] p The point to compute the distance from
/// @return (point index, distance)
std::pair<int, double> compute_closest_point(const BoundingBoxTree& tree,
                                             const Eigen::Vector3d& p);

/// Check whether point (x) is in bounding box
/// @param[in] b The bounding box
/// @param[in] x The point to check
/// @param[in] rtol Relative tolerance for checking if x is inside the
///                 bounding box
/// @return (point index, distance)
bool point_in_bbox(const Eigen::Array<double, 2, 3, Eigen::RowMajor>& b,
                   const Eigen::Vector3d& x, double rtol = 1e-14);

/// Compute squared distance between point and bounding box wih index
/// "node". Returns zero if point is inside box.
double compute_squared_distance_bbox(
    const Eigen::Array<double, 2, 3, Eigen::RowMajor>& b,
    const Eigen::Vector3d& x);

/// Compute squared distance from a given point to the nearest point on
/// a cell (only first order convex cells are supported at this stage)
/// Uses the GJK algorithm, see geometry::gjk_vector for details.
/// @param[in] entity MeshEntity
/// @param[in] p Point
/// @return shortest squared distance from p to entity
double squared_distance(const mesh::MeshEntity& entity,
                        const Eigen::Vector3d& p);

} // namespace geometry
} // namespace dolfinx
