// Copyright (C) 2019-2021 Garth N. Wells and Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "BoundingBoxTree.h"
#include <array>
#include <cstdint>
#include <deque>
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

/// @brief Compute squared distance between point and bounding box.
///
/// @param[in] b Bounding box coordinates
/// @param[in] x A point
/// @return The shortest distance between the bounding box `b` and the
/// point `x`. Returns zero if `x` is inside box.
double compute_squared_distance_bbox(std::span<const double, 6> b,
                                     std::span<const double, 3> x);

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

namespace impl
{
//-----------------------------------------------------------------------------

/// Check whether bounding box is a leaf node
constexpr bool is_leaf(std::array<int, 2> bbox)
{
  // Leaf nodes are marked by setting child_0 equal to child_1
  return bbox[0] == bbox[1];
}
//-----------------------------------------------------------------------------

/// A point `x` is inside a bounding box `b` if each component of its
/// coordinates lies within the range `[b(0,i), b(1,i)]` that defines
/// the bounds of the bounding box, b(0,i) <= x[i] <= b(1,i) for i = 0,
/// 1, 2
constexpr bool point_in_bbox(const std::array<double, 6>& b,
                             std::span<const double, 3> x)
{
  assert(b.size() == 6);
  constexpr double rtol = 1e-14;
  bool in = true;
  for (std::size_t i = 0; i < 3; i++)
  {
    double eps = rtol * (b[i + 3] - b[i]);
    in &= x[i] >= (b[i] - eps);
    in &= x[i] <= (b[i + 3] + eps);
  }

  return in;
}
//-----------------------------------------------------------------------------

/// A bounding box "a" is contained inside another bounding box "b", if
/// each  of its intervals [a(0,i), a(1,i)] is contained in [b(0,i),
/// b(1,i)], a(0,i) <= b(1, i) and a(1,i) >= b(0, i)
constexpr bool bbox_in_bbox(std::span<const double, 6> a,
                            std::span<const double, 6> b)
{
  constexpr double rtol = 1e-14;
  auto a0 = a.subspan<0, 3>();
  auto a1 = a.subspan<3, 3>();
  auto b0 = b.subspan<0, 3>();
  auto b1 = b.subspan<3, 3>();

  bool in = true;
  for (std::size_t i = 0; i < 3; i++)
  {
    double eps = rtol * (b1[i] - b0[i]);
    in &= a1[i] >= (b0[i] - eps);
    in &= a0[i] <= (b1[i] + eps);
  }

  return in;
}
//-----------------------------------------------------------------------------
// Compute closest entity {closest_entity, R2} (recursive)
inline std::pair<std::int32_t, double>
_compute_closest_entity(const geometry::BoundingBoxTree<double>& tree,
                        std::span<const double, 3> point, std::int32_t node,
                        const mesh::Mesh<double>& mesh,
                        std::int32_t closest_entity, double R2)
{
  // Get children of current bounding box node (child_1 denotes entity
  // index for leaves)
  const std::array<int, 2> bbox = tree.bbox(node);
  double r2;
  if (is_leaf(bbox))
  {
    // If point cloud tree the exact distance is easy to compute
    if (tree.tdim() == 0)
    {
      std::array<double, 6> diff = tree.get_bbox(node);
      for (std::size_t k = 0; k < 3; ++k)
        diff[k] -= point[k];
      r2 = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
    }
    else
    {
      r2 = geometry::compute_squared_distance_bbox(tree.get_bbox(node), point);

      // If bounding box closer than previous closest entity, use gjk to
      // obtain exact distance to the convex hull of the entity
      if (r2 <= R2)
      {
        r2 = geometry::squared_distance(
                 mesh, tree.tdim(), std::span(std::next(bbox.begin(), 1), 1),
                 point)
                 .front();
      }
    }

    // If entity is closer than best result so far, return it
    if (r2 <= R2)
    {
      closest_entity = bbox.back();
      R2 = r2;
    }

    return {closest_entity, R2};
  }
  else
  {
    // If bounding box is outside radius, then don't search further
    r2 = geometry::compute_squared_distance_bbox(tree.get_bbox(node), point);
    if (r2 > R2)
      return {closest_entity, R2};

    // Check both children
    // We use R2 (as opposed to r2), as a bounding box can be closer
    // than the actual entity
    std::pair<int, double> p0 = _compute_closest_entity(
        tree, point, bbox.front(), mesh, closest_entity, R2);
    std::pair<int, double> p1 = _compute_closest_entity(
        tree, point, bbox.back(), mesh, p0.first, p0.second);
    return p1;
  }
}
//-----------------------------------------------------------------------------
/// Compute collisions with a single point
/// @param[in] tree The bounding box tree
/// @param[in] points The points (shape=(num_points, 3))
/// @param[in, out] entities The list of colliding entities (local to process)
inline void
_compute_collisions_point(const geometry::BoundingBoxTree<double>& tree,
                          std::span<const double, 3> p,
                          std::vector<std::int32_t>& entities)
{
  std::deque<std::int32_t> stack;
  std::int32_t next = tree.num_bboxes() - 1;
  while (next != -1)
  {
    const std::array<int, 2> bbox = tree.bbox(next);
    next = -1;

    if (is_leaf(bbox))
    {
      // If box is a leaf node then add it to the list of colliding entities
      entities.push_back(bbox[1]);
    }
    else
    {
      // Check whether the point collides with child nodes (left and right)
      bool left = point_in_bbox(tree.get_bbox(bbox[0]), p);
      bool right = point_in_bbox(tree.get_bbox(bbox[1]), p);
      if (left && right)
      {
        // If the point collides with both child nodes, add the right node to
        // the stack (for later visiting) and continue the tree traversal with
        // the left subtree
        stack.push_back(bbox[1]);
        next = bbox[0];
      }
      else if (left)
      {
        // Traverse the current node's left subtree
        next = bbox[0];
      }
      else if (right)
      {
        // Traverse the current node's right subtree
        next = bbox[1];
      }
    }
    // If tree traversal reaches a dead end (box is a leaf node or no collision
    // detected), check the stack for deferred subtrees.
    if (next == -1 and !stack.empty())
    {
      next = stack.back();
      stack.pop_back();
    }
  }
}
//-----------------------------------------------------------------------------
// Compute collisions with tree (recursive)
inline void _compute_collisions_tree(const geometry::BoundingBoxTree<double>& A,
                                     const geometry::BoundingBoxTree<double>& B,
                                     std::int32_t node_A, std::int32_t node_B,
                                     std::vector<std::int32_t>& entities)
{
  // If bounding boxes don't collide, then don't search further
  if (!bbox_in_bbox(A.get_bbox(node_A), B.get_bbox(node_B)))
    return;

  // Get bounding boxes for current nodes
  const std::array<std::int32_t, 2> bbox_A = A.bbox(node_A);
  const std::array<std::int32_t, 2> bbox_B = B.bbox(node_B);

  // Check whether we've reached a leaf in A or B
  const bool is_leaf_A = is_leaf(bbox_A);
  const bool is_leaf_B = is_leaf(bbox_B);
  if (is_leaf_A and is_leaf_B)
  {
    // If both boxes are leaves (which we know collide), then add them
    // child_1 denotes entity for leaves
    entities.push_back(bbox_A[1]);
    entities.push_back(bbox_B[1]);
  }
  else if (is_leaf_A)
  {
    // If we reached the leaf in A, then descend B
    _compute_collisions_tree(A, B, node_A, bbox_B[0], entities);
    _compute_collisions_tree(A, B, node_A, bbox_B[1], entities);
  }
  else if (is_leaf_B)
  {
    // If we reached the leaf in B, then descend A
    _compute_collisions_tree(A, B, bbox_A[0], node_B, entities);
    _compute_collisions_tree(A, B, bbox_A[1], node_B, entities);
  }
  else if (node_A > node_B)
  {
    // At this point, we know neither is a leaf so descend the largest
    // tree first. Note that nodes are added in reverse order with the
    // top bounding box at the end so the largest tree (the one with the
    // the most boxes left to traverse) has the largest node number.
    _compute_collisions_tree(A, B, bbox_A[0], node_B, entities);
    _compute_collisions_tree(A, B, bbox_A[1], node_B, entities);
  }
  else
  {
    _compute_collisions_tree(A, B, node_A, bbox_B[0], entities);
    _compute_collisions_tree(A, B, node_A, bbox_B[1], entities);
  }

  // Note that cases above can be collected in fewer cases but this way
  // the logic is easier to follow.
}
//-----------------------------------------------------------------------------

} // namespace impl

/// @brief Create a bounding box tree for the midpoints of a subset of
/// entities.
/// @param[in] mesh The mesh
/// @param[in] tdim The topological dimension of the entity
/// @param[in] entity_indices List of local entity indices
/// @return Bounding box tree for midpoints of entities
BoundingBoxTree<double>
create_midpoint_tree(const mesh::Mesh<double>& mesh, int tdim,
                     std::span<const std::int32_t> entity_indices);

/// @brief Compute all collisions between two bounding box trees.
/// @param[in] tree0 First BoundingBoxTree
/// @param[in] tree1 Second BoundingBoxTree
/// @return List of pairs of intersecting box indices from each tree,
/// flattened as a vector of size num_intersections*2
std::vector<std::int32_t>
compute_collisions(const BoundingBoxTree<double>& tree0,
                   const BoundingBoxTree<double>& tree1);

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
compute_collisions(const BoundingBoxTree<double>& tree,
                   std::span<const double> points);

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
                                 const BoundingBoxTree<double>& tree,
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
std::vector<std::int32_t>
compute_closest_entity(const BoundingBoxTree<double>& tree,
                       const BoundingBoxTree<double>& midpoint_tree,
                       const mesh::Mesh<double>& mesh,
                       std::span<const double> points);
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
