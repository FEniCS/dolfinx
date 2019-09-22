// Copyright (C) 2019 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <utility>
#include <vector>

namespace dolfin
{
namespace mesh
{
class Mesh;
class MeshEntity;
} // namespace mesh

namespace geometry
{
class BoundingBoxTree;

// FIXME: document properly
/// Compute all collisions between bounding boxes and BoundingBoxTree
std::pair<std::vector<int>, std::vector<int>>
compute_collisions(const BoundingBoxTree& tree0, const BoundingBoxTree& tree1);

/// Compute all collisions between entities and BoundingBoxTree
std::pair<std::vector<int>, std::vector<int>>
compute_entity_collisions(const BoundingBoxTree& tree0,
                          const BoundingBoxTree& tree1, const mesh::Mesh& mesh0,
                          const mesh::Mesh& mesh1);

/// Compute all collisions between bounding boxes and point
/// @param[in] tree The bounding box tree
/// @param[in] p The point
/// @return Bounding box leaves that contain the point
std::vector<int> compute_collisions(const BoundingBoxTree& tree,
                                    const Eigen::Vector3d& p);

/// Compute all collisions between mesh entities and point
/// @param[in] tree The bounding box tree
/// @param[in] p The point
/// @param[in] mesh The mesh
/// @return Mesh entities that contain the point
std::vector<int> compute_entity_collisions(const BoundingBoxTree& tree,
                                           const Eigen::Vector3d& p,
                                           const mesh::Mesh& mesh);

/// Compute first collision between bounding boxes and Point
/// @param[in] tree The bounding box tree
/// @param[in] p The point
/// @return Index of the first found box that contains the point
int compute_first_collision(const BoundingBoxTree& tree,
                            const Eigen::Vector3d& p);

/// Compute first collision between entities and point
/// @param[in] tree The bounding box tree
/// @param[in] p The point
/// @param[in] mesh The mesh
/// @return Index of the first found mesh entity that contains the point
int compute_first_entity_collision(const BoundingBoxTree& tree,
                                   const Eigen::Vector3d& p,
                                   const mesh::Mesh& mesh);

/// Compute all collisions between processes and Point returning a
/// list of process ranks
std::vector<int> compute_process_collisions(const BoundingBoxTree& tree,
                                            const Eigen::Vector3d& p);

/// Compute squared distance from a given point to the nearest point on
/// a cell (only simplex cells are supported at this stage)
double squared_distance(const mesh::MeshEntity& entity,
                        const Eigen::Vector3d& p);

/// Compute squared distance to given point. This version takes the
/// three vertex coordinates as 3D points. This makes it possible to
/// reuse this function for computing the (squared) distance to a
/// tetrahedron.
double squared_distance_triangle(const Eigen::Vector3d& point,
                                 const Eigen::Vector3d& a,
                                 const Eigen::Vector3d& b,
                                 const Eigen::Vector3d& c);

/// Compute squared distance to given point. This version takes the two
/// vertex coordinates as 3D points. This makes it possible to reuse
/// this function for computing the (squared) distance to a triangle.
double squared_distance_interval(const Eigen::Vector3d& point,
                                 const Eigen::Vector3d& a,
                                 const Eigen::Vector3d& b);

} // namespace geometry
} // namespace dolfin