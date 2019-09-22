// Copyright (C) 2019 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
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

/// Compute all collisions between bounding boxes and Point
std::vector<int> compute_collisions(const BoundingBoxTree& tree,
                                    const Eigen::Vector3d& point);

/// Compute all collisions between entities and Point
std::vector<int> compute_entity_collisions(const BoundingBoxTree& tree,
                                           const Eigen::Vector3d& point,
                                           const mesh::Mesh& mesh);

/// Compute first collision between bounding boxes and Point
int compute_first_collision(const BoundingBoxTree& tree,
                            const Eigen::Vector3d& point);

/// Compute first collision between entities and Point
int compute_first_entity_collision(const BoundingBoxTree& tree,
                                   const Eigen::Vector3d& point,
                                   const mesh::Mesh& mesh);

/// Determine if a point collides with a BoundingBox of the tree
bool collides(const BoundingBoxTree& tree, const Eigen::Vector3d& point);

/// Determine if a point collides with an entity of the mesh (usually
/// a cell)
bool collides_entity(const BoundingBoxTree& tree, const Eigen::Vector3d& point,
                     const mesh::Mesh& mesh);

/// Compute squared distance from a given point to the nearest point on
/// a cell (only simplex cells are supported at this stage)
double squared_distance(const mesh::MeshEntity& entity,
                        const Eigen::Vector3d& point);

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