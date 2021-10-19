// Copyright (C) 2019-2021 Garth N. Wells and Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <dolfinx/graph/AdjacencyList.h>
#include <utility>
#include <vector>
#include <xtensor/xfixed.hpp>
#include <xtensor/xshape.hpp>
#include <xtensor/xtensor.hpp>
#include <xtl/xspan.hpp>

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
                     const xtl::span<const std::int32_t>& entity_indices);

/// Compute all collisions between two BoundingBoxTrees (local to
/// process)
/// @param[in] tree0 First BoundingBoxTree
/// @param[in] tree1 Second BoundingBoxTree
/// @return List of pairs of intersecting box indices (local to process)
/// from each tree
std::vector<std::array<int, 2>>
compute_collisions(const BoundingBoxTree& tree0, const BoundingBoxTree& tree1);

/// Compute all collisions between bounding boxes and for a set of
/// points
/// @param[in] tree The bounding box tree
/// @param[in] points The points (shape=(num_points, 3))
/// @return An adjacency list where the ith link corresponds to the
/// bounding box leaves (local to process) that contains the ith point
dolfinx::graph::AdjacencyList<std::int32_t>
compute_collisions(const BoundingBoxTree& tree,
                   const xt::xtensor<double, 2>& points);

/// Compute closest mesh entity (local to process) for the topological
/// distance of the bounding box tree and distance and a point
/// @param[in] tree The bounding box tree for the entities
/// @param[in] midpoint_tree A bounding box tree with the midpoints of
/// all the mesh entities
/// @param[in] points The set of points (shape=(num_points, 3))
/// @param[in] mesh The mesh
/// @return List of closest entities (local to process) where the ith
/// entry corresponds to the ith input point
/// @note Returns entity index -1 if the bounding box tree is empty
std::vector<std::int32_t> compute_closest_entity(
    const BoundingBoxTree& tree, const BoundingBoxTree& midpoint_tree,
    const xt::xtensor<double, 2>& points, const mesh::Mesh& mesh);

/// Compute squared distance between point and bounding box wih index
/// "node". Returns zero if point is inside box.
double compute_squared_distance_bbox(
    const xt::xtensor_fixed<double, xt::xshape<2, 3>>& b,
    const xt::xtensor_fixed<double, xt::xshape<3>>& x);

/// Compute the shortest vector from a set of points to a set of mesh
/// entities (local to process). The shortest vector is computed between
/// the ith row in the input points and the ith entry in the input
/// entities.
/// @param[in] mesh The mesh
/// @param[in] dim The topological dimension of the mesh entity
/// @param[in] entities The list of entities (local to process)
/// @param[in] points The set of points, shape (num_points, 3)
/// @return A two dimensional array of vectors, where the ith row
/// corresponds to the shortest vector between the between the ith
/// entity in entities and the ith row in points
xt::xtensor<double, 2>
shortest_vector(const mesh::Mesh& mesh, int dim,
                const xtl::span<const std::int32_t>& entities,
                const xt::xtensor<double, 2>& points);

/// Compute the squared distances from a set of points to a set of mesh
/// entities. The distance is computed between the ith input points and
/// the ith input entity.
/// @note Uses the GJK algorithm, see geometry::compute_distance_gjk for
/// details.
/// @note Uses a convex hull approximation of linearized geometry
/// @param[in] mesh Mesh containing the entities
/// @param[in] dim The topological dimension of the mesh entities
/// @param[in] entities The indices of the mesh entities (local to process)
/// @param[in] points The set points from which to computed the shortest
/// distance between the ith point and ith entity (shape=(num_points, 3))
/// @return Squared distances from points[i] to entities[i]
xt::xtensor<double, 1>
squared_distance(const mesh::Mesh& mesh, int dim,
                 const xtl::span<const std::int32_t>& entities,
                 const xt::xtensor<double, 2>& points);

/// From the given Mesh, find which cells actually collide with a set of
/// points. For the ith point, we supply a set of candidate cells.
/// @note Uses the GJK algorithm, see geometry::compute_distance_gjk for
/// details.
/// @param[in] mesh Mesh
/// @param[in] candidate_cells Adjacency list where the ith node
/// corresponds to the possible colliding entities with the ith point
/// @param[in] points The points to check for collision
/// (shape=(num_points, 3))
/// @return Adjacency list where the ith node is the list of entities
/// (local to process) that collide with the ith point
/// @note There may be nodes with no entries in the adjacency list
dolfinx::graph::AdjacencyList<int> select_colliding_cells(
    const dolfinx::mesh::Mesh& mesh,
    const dolfinx::graph::AdjacencyList<std::int32_t>& candidate_cells,
    const xt::xtensor<double, 2>& points);
} // namespace dolfinx::geometry
