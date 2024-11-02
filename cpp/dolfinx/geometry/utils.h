// Copyright (C) 2019-2021 Garth N. Wells and Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "BoundingBoxTree.h"
#include "gjk.h"
#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <deque>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <map>
#include <numeric>
#include <span>
#include <vector>

namespace dolfinx::geometry
{
/// @brief Information on the ownership of points distributed across
/// processes.
/// @tparam T Mesh geometry floating type.
template <std::floating_point T>
struct PointOwnershipData
{
  std::vector<int> src_owner; ///<  Ranks owning each point sent into ownership
                              ///<  determination for current process
  std::vector<int>
      dest_owners; ///< Ranks that sent `dest_points` to current process
  std::vector<T> dest_points; ///< Points that are owned by current process
  std::vector<std::int32_t>
      dest_cells; ///< Cell indices (local to process) where each entry of
                  ///< `dest_points` is located
};

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
template <std::floating_point T>
std::vector<T> shortest_vector(const mesh::Mesh<T>& mesh, int dim,
                               std::span<const std::int32_t> entities,
                               std::span<const T> points)
{
  const int tdim = mesh.topology()->dim();
  const mesh::Geometry<T>& geometry = mesh.geometry();

  std::span<const T> geom_dofs = geometry.x();
  auto x_dofmap = geometry.dofmap();
  std::vector<T> shortest_vectors;
  shortest_vectors.reserve(3 * entities.size());
  if (dim == tdim)
  {
    for (std::size_t e = 0; e < entities.size(); e++)
    {
      // Check that we have sent in valid entities, i.e. that they exist in the
      // local dofmap. One gets a cryptical memory segfault if entities is -1
      assert(entities[e] >= 0);
      auto dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          x_dofmap, entities[e], MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      std::vector<T> nodes(3 * dofs.size());
      for (std::size_t i = 0; i < dofs.size(); ++i)
      {
        const std::int32_t pos = 3 * dofs[i];
        for (std::size_t j = 0; j < 3; ++j)
          nodes[3 * i + j] = geom_dofs[pos + j];
      }

      std::array<T, 3> d
          = compute_distance_gjk<T>(points.subspan(3 * e, 3), nodes);
      shortest_vectors.insert(shortest_vectors.end(), d.begin(), d.end());
    }
  }
  else
  {
    mesh.topology_mutable()->create_connectivity(dim, tdim);
    mesh.topology_mutable()->create_connectivity(tdim, dim);
    auto e_to_c = mesh.topology()->connectivity(dim, tdim);
    assert(e_to_c);
    auto c_to_e = mesh.topology_mutable()->connectivity(tdim, dim);
    assert(c_to_e);
    for (std::size_t e = 0; e < entities.size(); e++)
    {
      const std::int32_t index = entities[e];

      // Find attached cell
      assert(e_to_c->num_links(index) > 0);
      const std::int32_t c = e_to_c->links(index)[0];

      // Find local number of entity wrt cell
      auto cell_entities = c_to_e->links(c);
      auto it0 = std::find(cell_entities.begin(), cell_entities.end(), index);
      assert(it0 != cell_entities.end());
      const int local_cell_entity = std::distance(cell_entities.begin(), it0);

      // Tabulate geometry dofs for the entity
      auto dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          x_dofmap, c, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      const std::vector<int> entity_dofs
          = geometry.cmap().create_dof_layout().entity_closure_dofs(
              dim, local_cell_entity);
      std::vector<T> nodes(3 * entity_dofs.size());
      for (std::size_t i = 0; i < entity_dofs.size(); i++)
      {
        const std::int32_t pos = 3 * dofs[entity_dofs[i]];
        for (std::size_t j = 0; j < 3; ++j)
          nodes[3 * i + j] = geom_dofs[pos + j];
      }

      std::array<T, 3> d
          = compute_distance_gjk<T>(points.subspan(3 * e, 3), nodes);
      shortest_vectors.insert(shortest_vectors.end(), d.begin(), d.end());
    }
  }

  return shortest_vectors;
}

/// @brief Compute squared distance between point and bounding box.
///
/// @param[in] b Bounding box coordinates
/// @param[in] x A point
/// @return The shortest distance between the bounding box `b` and the
/// point `x`. Returns zero if `x` is inside box.
template <std::floating_point T>
T compute_squared_distance_bbox(std::span<const T, 6> b,
                                std::span<const T, 3> x)
{
  auto b0 = b.template subspan<0, 3>();
  auto b1 = b.template subspan<3, 3>();
  return std::transform_reduce(x.begin(), x.end(), b0.begin(), 0.0,
                               std::plus<>{},
                               [](auto x, auto b)
                               {
                                 auto dx = x - b;
                                 return dx > 0 ? 0 : dx * dx;
                               })
         + std::transform_reduce(x.begin(), x.end(), b1.begin(), 0.0,
                                 std::plus<>{},
                                 [](auto x, auto b)
                                 {
                                   auto dx = x - b;
                                   return dx < 0 ? 0 : dx * dx;
                                 });
}

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
template <std::floating_point T>
std::vector<T> squared_distance(const mesh::Mesh<T>& mesh, int dim,
                                std::span<const std::int32_t> entities,
                                std::span<const T> points)
{
  std::vector<T> v = shortest_vector(mesh, dim, entities, points);
  std::vector<T> d(v.size() / 3, 0);
  for (std::size_t i = 0; i < d.size(); ++i)
    for (std::size_t j = 0; j < 3; ++j)
      d[i] += v[3 * i + j] * v[3 * i + j];
  return d;
}

namespace impl
{
/// Check whether bounding box is a leaf node
constexpr bool is_leaf(std::array<int, 2> bbox)
{
  // Leaf nodes are marked by setting child_0 equal to child_1
  return bbox[0] == bbox[1];
}

/// A point `x` is inside a bounding box `b` if each component of its
/// coordinates lies within the range `[b(0,i), b(1,i)]` that defines
/// the bounds of the bounding box, b(0,i) <= x[i] <= b(1,i) for i = 0,
/// 1, 2
template <std::floating_point T>
constexpr bool point_in_bbox(const std::array<T, 6>& b, std::span<const T, 3> x)
{
  constexpr T rtol = 1e-14;
  bool in = true;
  for (std::size_t i = 0; i < 3; i++)
  {
    T eps = rtol * (b[i + 3] - b[i]);
    in &= x[i] >= (b[i] - eps);
    in &= x[i] <= (b[i + 3] + eps);
  }

  return in;
}

/// A bounding box "a" is contained inside another bounding box "b", if
/// each  of its intervals [a(0,i), a(1,i)] is contained in [b(0,i),
/// b(1,i)], a(0,i) <= b(1, i) and a(1,i) >= b(0, i)
template <std::floating_point T>
constexpr bool bbox_in_bbox(std::span<const T, 6> a, std::span<const T, 6> b)
{
  constexpr T rtol = 1e-14;
  auto a0 = a.template subspan<0, 3>();
  auto a1 = a.template subspan<3, 3>();
  auto b0 = b.template subspan<0, 3>();
  auto b1 = b.template subspan<3, 3>();

  bool in = true;
  for (std::size_t i = 0; i < 3; i++)
  {
    T eps = rtol * (b1[i] - b0[i]);
    in &= a1[i] >= (b0[i] - eps);
    in &= a0[i] <= (b1[i] + eps);
  }

  return in;
}

/// Compute closest entity {closest_entity, R2} (recursive)
template <std::floating_point T>
std::pair<std::int32_t, T>
_compute_closest_entity(const geometry::BoundingBoxTree<T>& tree,
                        std::span<const T, 3> point, std::int32_t node,
                        const mesh::Mesh<T>& mesh, std::int32_t closest_entity,
                        T R2)
{
  // Get children of current bounding box node (child_1 denotes entity
  // index for leaves)
  const std::array<int, 2> bbox = tree.bbox(node);
  T r2;
  if (is_leaf(bbox))
  {
    // If point cloud tree the exact distance is easy to compute
    if (tree.tdim() == 0)
    {
      std::array<T, 6> diff = tree.get_bbox(node);
      for (std::size_t k = 0; k < 3; ++k)
        diff[k] -= point[k];
      r2 = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
    }
    else
    {
      r2 = compute_squared_distance_bbox<T>(tree.get_bbox(node), point);

      // If bounding box closer than previous closest entity, use gjk to
      // obtain exact distance to the convex hull of the entity
      if (r2 <= R2)
      {
        r2 = squared_distance<T>(mesh, tree.tdim(),
                                 std::span(std::next(bbox.begin(), 1), 1),
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
    r2 = compute_squared_distance_bbox<T>(tree.get_bbox(node), point);
    if (r2 > R2)
      return {closest_entity, R2};

    // Check both children
    // We use R2 (as opposed to r2), as a bounding box can be closer
    // than the actual entity
    std::pair<int, T> p0 = _compute_closest_entity(tree, point, bbox.front(),
                                                   mesh, closest_entity, R2);
    std::pair<int, T> p1 = _compute_closest_entity(tree, point, bbox.back(),
                                                   mesh, p0.first, p0.second);
    return p1;
  }
}

/// @brief Compute collisions with a single point.
/// @param[in] tree The bounding box tree
/// @param[in] points The points (`shape=(num_points, 3)`)
/// @param[in, out] entities The list of colliding entities (local to
/// process)
template <std::floating_point T>
void _compute_collisions_point(const geometry::BoundingBoxTree<T>& tree,
                               std::span<const T, 3> p,
                               std::vector<std::int32_t>& entities)
{
  std::deque<std::int32_t> stack;
  std::int32_t next = tree.num_bboxes() - 1;
  while (next != -1)
  {
    const std::array<int, 2> bbox = tree.bbox(next);
    if (is_leaf(bbox) and point_in_bbox(tree.get_bbox(next), p))
    {
      // If box is a leaf node then add it to the list of colliding
      // entities
      entities.push_back(bbox[1]);
      next = -1;
    }
    else
    {
      // Check whether the point collides with child nodes (left and
      // right)
      bool left = point_in_bbox(tree.get_bbox(bbox[0]), p);
      bool right = point_in_bbox(tree.get_bbox(bbox[1]), p);
      if (left and right)
      {
        // If the point collides with both child nodes, add the right
        // node to the stack (for later visiting) and continue the tree
        // traversal with the left subtree
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
      else
        next = -1;
    }

    // If tree traversal reaches a dead end (box is a leaf node or no
    // collision detected), check the stack for deferred subtrees
    if (next == -1 and !stack.empty())
    {
      next = stack.back();
      stack.pop_back();
    }
  }
}

// Compute collisions with tree (recursive)
template <std::floating_point T>
void _compute_collisions_tree(const geometry::BoundingBoxTree<T>& A,
                              const geometry::BoundingBoxTree<T>& B,
                              std::int32_t node_A, std::int32_t node_B,
                              std::vector<std::int32_t>& entities)
{
  // If bounding boxes don't collide, then don't search further
  if (!bbox_in_bbox<T>(A.get_bbox(node_A), B.get_bbox(node_B)))
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

} // namespace impl

/// @brief Create a bounding box tree for the midpoints of a subset of
/// entities.
/// @param[in] mesh The mesh
/// @param[in] tdim The topological dimension of the entity
/// @param[in] entities List of local entity indices
/// @return Bounding box tree for midpoints of entities
template <std::floating_point T>
BoundingBoxTree<T> create_midpoint_tree(const mesh::Mesh<T>& mesh, int tdim,
                                        std::span<const std::int32_t> entities)
{
  spdlog::info("Building point search tree to accelerate distance queries for "
               "a given topological dimension and subset of entities.");

  const std::vector<T> midpoints
      = mesh::compute_midpoints(mesh, tdim, entities);
  std::vector<std::pair<std::array<T, 3>, std::int32_t>> points(
      entities.size());
  for (std::size_t i = 0; i < points.size(); ++i)
  {
    for (std::size_t j = 0; j < 3; ++j)
      points[i].first[j] = midpoints[3 * i + j];
    points[i].second = entities[i];
  }

  // Build tree
  return BoundingBoxTree(points);
}

/// @brief Compute all collisions between two bounding box trees.
/// @param[in] tree0 First BoundingBoxTree
/// @param[in] tree1 Second BoundingBoxTree
/// @return List of pairs of intersecting box indices from each tree,
/// flattened as a vector of size num_intersections*2
template <std::floating_point T>
std::vector<std::int32_t> compute_collisions(const BoundingBoxTree<T>& tree0,
                                             const BoundingBoxTree<T>& tree1)
{
  // Call recursive find function
  std::vector<std::int32_t> entities;
  if (tree0.num_bboxes() > 0 and tree1.num_bboxes() > 0)
  {
    impl::_compute_collisions_tree(tree0, tree1, tree0.num_bboxes() - 1,
                                   tree1.num_bboxes() - 1, entities);
  }

  return entities;
}

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
template <std::floating_point T>
graph::AdjacencyList<std::int32_t>
compute_collisions(const BoundingBoxTree<T>& tree, std::span<const T> points)
{
  if (tree.num_bboxes() > 0)
  {
    std::vector<std::int32_t> entities, offsets(points.size() / 3 + 1, 0);
    entities.reserve(points.size() / 3);
    for (std::size_t p = 0; p < points.size() / 3; ++p)
    {
      impl::_compute_collisions_point(
          tree, std::span<const T, 3>(points.data() + 3 * p, 3), entities);
      offsets[p + 1] = entities.size();
    }

    return graph::AdjacencyList(std::move(entities), std::move(offsets));
  }
  else
  {
    return graph::AdjacencyList(
        std::vector<std::int32_t>(),
        std::vector<std::int32_t>(points.size() / 3 + 1, 0));
  }
}

/// @brief Given a set of cells, find the first one that collides with a
/// point.
///
/// A point can collide with more than one cell. The first cell detected
/// to collide with the point is returned. If no collision is detected,
/// -1 is returned.
///
/// @note `cells` can for instance be found by using
/// geometry::compute_collisions between a bounding box tree for the
/// cells of the mesh and the point.
///
/// @param[in] mesh The mesh.
/// @param[in] cells Candidate cells.
/// @param[in] point The point (`shape=(3,)`).
/// @param[in] tol Tolerance for accepting a collision (in the squared
/// distance).
/// @return Local cell index, -1 if not found.
template <std::floating_point T>
std::int32_t compute_first_colliding_cell(const mesh::Mesh<T>& mesh,
                                          std::span<const std::int32_t> cells,
                                          std::array<T, 3> point, T tol)
{
  if (cells.empty())
    return -1;
  else
  {
    const mesh::Geometry<T>& geometry = mesh.geometry();
    std::span<const T> geom_dofs = geometry.x();
    auto x_dofmap = geometry.dofmap();
    const std::size_t num_nodes = x_dofmap.extent(1);
    std::vector<T> coordinate_dofs(num_nodes * 3);
    for (auto cell : cells)
    {
      auto dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          x_dofmap, cell, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      for (std::size_t i = 0; i < num_nodes; ++i)
      {
        std::copy_n(std::next(geom_dofs.begin(), 3 * dofs[i]), 3,
                    std::next(coordinate_dofs.begin(), 3 * i));
      }

      std::array<T, 3> shortest_vector
          = compute_distance_gjk<T>(point, coordinate_dofs);
      T d2 = std::reduce(shortest_vector.begin(), shortest_vector.end(), T(0),
                         [](auto d, auto e) { return d + e * e; });
      if (d2 < tol)
        return cell;
    }

    return -1;
  }
}

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
template <std::floating_point T>
std::vector<std::int32_t>
compute_closest_entity(const BoundingBoxTree<T>& tree,
                       const BoundingBoxTree<T>& midpoint_tree,
                       const mesh::Mesh<T>& mesh, std::span<const T> points)
{
  if (tree.num_bboxes() == 0)
    return std::vector<std::int32_t>(points.size() / 3, -1);

  std::vector<std::int32_t> entities;
  entities.reserve(points.size() / 3);
  for (std::size_t i = 0; i < points.size() / 3; ++i)
  {
    // Use midpoint tree to find initial closest entity to the point.
    // Start by using a leaf node as the initial guess for the input
    // entity
    std::array<int, 2> leaf0 = midpoint_tree.bbox(0);
    assert(impl::is_leaf(leaf0));
    std::array<T, 6> diff = midpoint_tree.get_bbox(0);
    for (std::size_t k = 0; k < 3; ++k)
      diff[k] -= points[3 * i + k];
    T R2 = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];

    // Use a recursive search through the bounding box tree
    // to find determine the entity with the closest midpoint.
    // As the midpoint tree only consist of points, the distance
    // queries are lightweight.
    const auto [m_index, m_distance2] = impl::_compute_closest_entity(
        midpoint_tree, std::span<const T, 3>(points.data() + 3 * i, 3),
        midpoint_tree.num_bboxes() - 1, mesh, leaf0[0], R2);

    // Use a recursives search through the bounding box tree to
    // determine which entity is actually closest.
    // Uses the entity with the closest midpoint as initial guess, and
    // the distance from the midpoint to the point of interest as the
    // initial search radius.
    const auto [index, distance2] = impl::_compute_closest_entity(
        tree, std::span<const T, 3>(points.data() + 3 * i, 3),
        tree.num_bboxes() - 1, mesh, m_index, m_distance2);

    entities.push_back(index);
  }

  return entities;
}

/// @brief Compute which cells collide with a point.
///
/// @note Uses the GJK algorithm, see geometry::compute_distance_gjk for
/// details.
///
/// @note `candidate_cells` can for instance be found by using
/// geometry::compute_collisions between a bounding box tree and the set
/// of points.
///
/// @param[in] mesh The mesh
/// @param[in] candidate_cells List of candidate colliding cells for the
/// ith point in `points`
/// @param[in] points Points to check for collision (`shape=(num_points,
/// 3)`). Storage is row-major.
/// @return For each point, the cells that collide with the point.
template <std::floating_point T>
graph::AdjacencyList<std::int32_t> compute_colliding_cells(
    const mesh::Mesh<T>& mesh,
    const graph::AdjacencyList<std::int32_t>& candidate_cells,
    std::span<const T> points)
{
  std::vector<std::int32_t> offsets = {0};
  offsets.reserve(candidate_cells.num_nodes() + 1);
  std::vector<std::int32_t> colliding_cells;
  constexpr T eps2 = 1e-12;
  const int tdim = mesh.topology()->dim();
  for (std::int32_t i = 0; i < candidate_cells.num_nodes(); i++)
  {
    auto cells = candidate_cells.links(i);
    std::vector<T> _point(3 * cells.size());
    for (std::size_t j = 0; j < cells.size(); ++j)
      for (std::size_t k = 0; k < 3; ++k)
        _point[3 * j + k] = points[3 * i + k];

    std::vector distances_sq = squared_distance<T>(mesh, tdim, cells, _point);
    for (std::size_t j = 0; j < cells.size(); j++)
      if (distances_sq[j] < eps2)
        colliding_cells.push_back(cells[j]);

    offsets.push_back(colliding_cells.size());
  }

  return graph::AdjacencyList(std::move(colliding_cells), std::move(offsets));
}

/// @brief Given a set of points, determine which process is colliding,
/// using the GJK algorithm on cells to determine collisions.
///
/// @todo This docstring is unclear. Needs fixing.
///
/// @param[in] mesh The mesh
/// @param[in] points Points to check for collision (`shape=(num_points,
/// 3)`). Storage is row-major.
/// @param[in] padding Amount of absolute padding of bounding boxes of the mesh.
/// Each bounding box of the mesh is padded with this amount, to increase
/// the number of candidates, avoiding rounding errors in determining the owner
/// of a point if the point is on the surface of a cell in the mesh.
/// @return Tuple `(src_owner, dest_owner, dest_points, dest_cells)`,
/// where src_owner is a list of ranks corresponding to the input
/// points. dest_owner is a list of ranks corresponding to dest_points,
/// the points that this process owns. dest_cells contains the
/// corresponding cell for each entry in dest_points.
///
/// @note `dest_owner` is sorted
/// @note Returns -1 if no colliding process is found
/// @note dest_points is flattened row-major, shape `(dest_owner.size(),
/// 3)`
/// @note Only looks through cells owned by the process
/// @note A large padding value can increase the runtime of the function by
/// orders of magnitude, because for non-colliding cells
/// one has to determine the closest cell among all processes with an
/// intersecting bounding box, which is an expensive operation to perform.
template <std::floating_point T>
PointOwnershipData<T> determine_point_ownership(const mesh::Mesh<T>& mesh,
                                                std::span<const T> points,
                                                T padding)
{
  MPI_Comm comm = mesh.comm();

  // Create a global bounding-box tree to find candidate processes with
  // cells that could collide with the points
  const int tdim = mesh.topology()->dim();
  auto cell_map = mesh.topology()->index_map(tdim);
  const std::int32_t num_cells = cell_map->size_local();
  // NOTE: Should we send the cells in as input?
  std::vector<std::int32_t> cells(num_cells, 0);
  std::iota(cells.begin(), cells.end(), 0);
  BoundingBoxTree bb(mesh, tdim, cells, padding);
  BoundingBoxTree global_bbtree = bb.create_global_tree(comm);

  // Compute collisions:
  // For each point in `points` get the processes it should be sent to
  graph::AdjacencyList collisions = compute_collisions(global_bbtree, points);

  // Get unique list of outgoing ranks
  std::vector<std::int32_t> out_ranks = collisions.array();
  std::ranges::sort(out_ranks);
  auto [unique_end, range_end] = std::ranges::unique(out_ranks);
  out_ranks.erase(unique_end, range_end);

  // Compute incoming edges (source processes)
  std::vector in_ranks = dolfinx::MPI::compute_graph_edges_nbx(comm, out_ranks);
  std::ranges::sort(in_ranks);

  // Create neighborhood communicator in forward direction
  MPI_Comm forward_comm;
  MPI_Dist_graph_create_adjacent(
      comm, in_ranks.size(), in_ranks.data(), MPI_UNWEIGHTED, out_ranks.size(),
      out_ranks.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &forward_comm);

  // Compute map from global mpi rank to neighbor rank, "collisions"
  // uses global rank
  std::map<std::int32_t, std::int32_t> rank_to_neighbor;
  for (std::size_t i = 0; i < out_ranks.size(); i++)
    rank_to_neighbor[out_ranks[i]] = i;

  // Count the number of points to send per neighbor process
  std::vector<std::int32_t> send_sizes(out_ranks.size());
  for (std::size_t i = 0; i < points.size() / 3; ++i)
    for (auto p : collisions.links(i))
      send_sizes[rank_to_neighbor[p]] += 3;

  // Compute receive sizes
  std::vector<std::int32_t> recv_sizes(in_ranks.size());
  send_sizes.reserve(1);
  recv_sizes.reserve(1);
  MPI_Request sizes_request;
  MPI_Ineighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1,
                         MPI_INT, forward_comm, &sizes_request);

  // Compute sending offsets
  std::vector<std::int32_t> send_offsets(send_sizes.size() + 1, 0);
  std::partial_sum(send_sizes.begin(), send_sizes.end(),
                   std::next(send_offsets.begin(), 1));

  // Pack data to send and store unpack map
  std::vector<T> send_data(send_offsets.back());
  std::vector<std::int32_t> counter(send_sizes.size(), 0);
  // unpack map: [index in adj list][pos in x]
  std::vector<std::int32_t> unpack_map(send_offsets.back() / 3);
  for (std::size_t i = 0; i < points.size(); i += 3)
  {
    for (auto p : collisions.links(i / 3))
    {
      int neighbor = rank_to_neighbor[p];
      int pos = send_offsets[neighbor] + counter[neighbor];
      auto it = std::next(send_data.begin(), pos);
      std::copy_n(std::next(points.begin(), i), 3, it);
      unpack_map[pos / 3] = i / 3;
      counter[neighbor] += 3;
    }
  }

  MPI_Wait(&sizes_request, MPI_STATUS_IGNORE);
  std::vector<std::int32_t> recv_offsets(in_ranks.size() + 1, 0);
  std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                   std::next(recv_offsets.begin(), 1));

  std::vector<T> received_points((std::size_t)recv_offsets.back());
  MPI_Neighbor_alltoallv(
      send_data.data(), send_sizes.data(), send_offsets.data(),
      dolfinx::MPI::mpi_t<T>(), received_points.data(), recv_sizes.data(),
      recv_offsets.data(), dolfinx::MPI::mpi_t<T>(), forward_comm);

  // Get mesh geometry for closest entity
  const mesh::Geometry<T>& geometry = mesh.geometry();
  std::span<const T> geom_dofs = geometry.x();
  auto x_dofmap = geometry.dofmap();

  // Compute candidate cells for collisions (and extrapolation)
  const graph::AdjacencyList<std::int32_t> candidate_collisions
      = compute_collisions(bb, std::span<const T>(received_points.data(),
                                                  received_points.size()));

  // Each process checks which points collide with a cell on the process
  const int rank = dolfinx::MPI::rank(comm);
  std::vector<std::int32_t> cell_indicator(received_points.size() / 3);
  std::vector<std::int32_t> closest_cells(received_points.size() / 3);
  for (std::size_t p = 0; p < received_points.size(); p += 3)
  {
    std::array<T, 3> point;
    std::copy_n(std::next(received_points.begin(), p), 3, point.begin());
    // Find first colliding cell among the cells with colliding bounding boxes
    const int colliding_cell = geometry::compute_first_colliding_cell(
        mesh, candidate_collisions.links(p / 3), point,
        10 * std::numeric_limits<T>::epsilon());
    // If a collding cell is found, store the rank of the current process
    // which will be sent back to the owner of the point
    cell_indicator[p / 3] = (colliding_cell >= 0) ? rank : -1;
    // Store the cell index for lookup once the owning processes has determined
    // the ownership of the point
    closest_cells[p / 3] = colliding_cell;
  }

  // Create neighborhood communicator in the reverse direction: send
  // back col to requesting processes
  MPI_Comm reverse_comm;
  MPI_Dist_graph_create_adjacent(
      comm, out_ranks.size(), out_ranks.data(), MPI_UNWEIGHTED, in_ranks.size(),
      in_ranks.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &reverse_comm);

  // Reuse sizes and offsets from first communication set
  // but divide by three
  {
    auto rescale = [](auto& x)
    { std::ranges::transform(x, x.begin(), [](auto e) { return (e / 3); }); };
    rescale(recv_sizes);
    rescale(recv_offsets);
    rescale(send_sizes);
    rescale(send_offsets);

    // The communication is reversed, so swap recv to send offsets
    std::swap(recv_sizes, send_sizes);
    std::swap(recv_offsets, send_offsets);
  }

  std::vector<std::int32_t> recv_ranks(recv_offsets.back());
  MPI_Neighbor_alltoallv(cell_indicator.data(), send_sizes.data(),
                         send_offsets.data(), MPI_INT32_T, recv_ranks.data(),
                         recv_sizes.data(), recv_offsets.data(), MPI_INT32_T,
                         reverse_comm);

  std::vector<int> point_owners(points.size() / 3, -1);
  for (std::size_t i = 0; i < unpack_map.size(); i++)
  {
    const std::int32_t pos = unpack_map[i];
    // Only insert new owner if no owner has previously been found
    if (recv_ranks[i] >= 0 && point_owners[pos] == -1)
      point_owners[pos] = recv_ranks[i];
  }

  // Create extrapolation marker for those points already sent to other
  // process
  std::vector<std::uint8_t> send_extrapolate(recv_offsets.back());
  for (std::int32_t i = 0; i < recv_offsets.back(); i++)
  {
    const std::int32_t pos = unpack_map[i];
    send_extrapolate[i] = point_owners[pos] == -1;
  }

  // Swap communication direction, to send extrapolation marker to other
  // processes
  std::swap(send_sizes, recv_sizes);
  std::swap(send_offsets, recv_offsets);
  std::vector<std::uint8_t> dest_extrapolate(recv_offsets.back());
  MPI_Neighbor_alltoallv(send_extrapolate.data(), send_sizes.data(),
                         send_offsets.data(), MPI_UINT8_T,
                         dest_extrapolate.data(), recv_sizes.data(),
                         recv_offsets.data(), MPI_UINT8_T, forward_comm);

  std::vector<T> squared_distances(received_points.size() / 3, -1);

  for (std::size_t i = 0; i < dest_extrapolate.size(); i++)
  {
    if (dest_extrapolate[i] == 1)
    {
      assert(closest_cells[i] == -1);
      std::array<T, 3> point;
      std::copy_n(std::next(received_points.begin(), 3 * i), 3, point.begin());

      // Find shortest distance among cells with colldiing bounding box
      T shortest_distance = std::numeric_limits<T>::max();
      std::int32_t closest_cell = -1;
      for (auto cell : candidate_collisions.links(i))
      {
        auto dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            x_dofmap, cell, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        std::vector<T> nodes(3 * dofs.size());
        for (std::size_t j = 0; j < dofs.size(); ++j)
        {
          const int pos = 3 * dofs[j];
          for (std::size_t k = 0; k < 3; ++k)
            nodes[3 * j + k] = geom_dofs[pos + k];
        }
        const std::array<T, 3> d = compute_distance_gjk<T>(
            std::span<const T>(point.data(), point.size()), nodes);
        if (T current_distance = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            current_distance < shortest_distance)
        {
          shortest_distance = current_distance;
          closest_cell = cell;
        }
      }
      closest_cells[i] = closest_cell;
      squared_distances[i] = shortest_distance;
    }
  }

  std::swap(recv_sizes, send_sizes);
  std::swap(recv_offsets, send_offsets);

  // Get distances from closest entity of points that were on the other process
  std::vector<T> recv_distances(recv_offsets.back());
  MPI_Neighbor_alltoallv(
      squared_distances.data(), send_sizes.data(), send_offsets.data(),
      dolfinx::MPI::mpi_t<T>(), recv_distances.data(), recv_sizes.data(),
      recv_offsets.data(), dolfinx::MPI::mpi_t<T>(), reverse_comm);

  // Update point ownership with extrapolation information
  std::vector<T> closest_distance(point_owners.size(),
                                  std::numeric_limits<T>::max());
  for (std::size_t i = 0; i < out_ranks.size(); i++)
  {
    for (std::int32_t j = recv_offsets[i]; j < recv_offsets[i + 1]; j++)
    {
      const std::int32_t pos = unpack_map[j];
      auto current_dist = recv_distances[j];
      // Update if closer than previous guess and was found
      if (auto d = closest_distance[pos];
          (current_dist > 0) and (current_dist < d))
      {
        point_owners[pos] = out_ranks[i];
        closest_distance[pos] = current_dist;
      }
    }
  }

  // Communication is reversed again to send dest ranks to all processes
  std::swap(send_sizes, recv_sizes);
  std::swap(send_offsets, recv_offsets);

  // Pack ownership data
  std::vector<std::int32_t> send_owners(send_offsets.back());
  std::ranges::fill(counter, 0);
  for (std::size_t i = 0; i < points.size() / 3; ++i)
  {
    for (auto p : collisions.links(i))
    {
      int neighbor = rank_to_neighbor[p];
      send_owners[send_offsets[neighbor] + counter[neighbor]++]
          = point_owners[i];
    }
  }

  // Send ownership info
  std::vector<std::int32_t> dest_ranks(recv_offsets.back());
  MPI_Neighbor_alltoallv(send_owners.data(), send_sizes.data(),
                         send_offsets.data(), MPI_INT32_T, dest_ranks.data(),
                         recv_sizes.data(), recv_offsets.data(), MPI_INT32_T,
                         forward_comm);

  // Unpack dest ranks if point owner is this rank
  std::vector<int> owned_recv_ranks;
  owned_recv_ranks.reserve(recv_offsets.back());
  std::vector<T> owned_recv_points;
  std::vector<std::int32_t> owned_recv_cells;
  for (std::size_t i = 0; i < in_ranks.size(); i++)
  {
    for (std::int32_t j = recv_offsets[i]; j < recv_offsets[i + 1]; j++)
    {
      if (rank == dest_ranks[j])
      {
        owned_recv_ranks.push_back(in_ranks[i]);
        owned_recv_points.insert(
            owned_recv_points.end(), std::next(received_points.cbegin(), 3 * j),
            std::next(received_points.cbegin(), 3 * (j + 1)));
        owned_recv_cells.push_back(closest_cells[j]);
      }
    }
  }

  MPI_Comm_free(&forward_comm);
  MPI_Comm_free(&reverse_comm);
  return PointOwnershipData<T>{.src_owner = std::move(point_owners),
                               .dest_owners = std::move(owned_recv_ranks),
                               .dest_points = std::move(owned_recv_points),
                               .dest_cells = std::move(owned_recv_cells)};
}

} // namespace dolfinx::geometry
