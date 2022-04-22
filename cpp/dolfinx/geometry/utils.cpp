// Copyright (C) 2006-2021 Chris N. Richardson, Anders Logg, Garth N. Wells and
// Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include "BoundingBoxTree.h"
#include "gjk.h"
#include <deque>
#include <dolfinx/common/log.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>
#include <xtensor/xfixed.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xview.hpp>

using namespace dolfinx;

namespace
{
//-----------------------------------------------------------------------------
// Check whether bounding box is a leaf node
constexpr bool is_leaf(const std::array<int, 2>& bbox)
{
  // Leaf nodes are marked by setting child_0 equal to child_1
  return bbox[0] == bbox[1];
}
//-----------------------------------------------------------------------------
/// A point `x` is inside a bounding box `b` if each component of its coordinates
/// lies within the range `[b(0,i), b(1,i)]` that defines the bounds of the
/// bounding box, b(0,i) <= x[i] <= b(1,i) for i = 0, 1, 2
bool point_in_bbox(const xt::xtensor_fixed<double, xt::xshape<2, 3>>& b,
                   const xt::xtensor_fixed<double, xt::xshape<3>>& x)
{
  constexpr double rtol = 1e-14;
  double eps;
  bool in = true;
  for (int i = 0; i < 3; i++)
  {
    eps = rtol * (b(1, i) - b(0, i));
    in &= x[i] >= (b(0, i) - eps);
    in &= x[i] <= (b(1, i) + eps);
  }

  return in;
}
//-----------------------------------------------------------------------------
/// A bounding box "a" is contained inside another bounding box "b", if each
/// of its intervals [a(0,i), a(1,i)] is contained in [b(0,i), b(1,i)],
/// a(0,i) <= b(1, i) and a(1,i) >= b(0, i)
bool bbox_in_bbox(const xt::xtensor_fixed<double, xt::xshape<2, 3>>& a,
                  const xt::xtensor_fixed<double, xt::xshape<2, 3>>& b)
{
  constexpr double rtol = 1e-14;
  double eps;
  bool in = true;

  for (int i = 0; i < 3; i++)
  {
    eps = rtol * (b(1, i) - b(0, i));
    in &= a(1, i) >= (b(0, i) - eps);
    in &= a(0, i) <= (b(1, i) + eps);
  }

  return in;
}
//-----------------------------------------------------------------------------
// Compute closest entity {closest_entity, R2} (recursive)
std::pair<std::int32_t, double>
_compute_closest_entity(const geometry::BoundingBoxTree& tree,
                        const xt::xtensor_fixed<double, xt::xshape<3>>& point,
                        int node, const mesh::Mesh& mesh,
                        std::int32_t closest_entity, double R2)
{
  // Get children of current bounding box node (child_1 denotes entity
  // index for leaves)
  const std::array bbox = tree.bbox(node);
  double r2;
  if (is_leaf(bbox))
  {
    // If point cloud tree the exact distance is easy to compute
    if (tree.tdim() == 0)
    {
      xt::xtensor_fixed<double, xt::xshape<3>> diff
          = xt::row(tree.get_bbox(node), 0);
      diff -= point;
      r2 = xt::norm_sq(diff)();
    }
    else
    {
      r2 = geometry::compute_squared_distance_bbox(tree.get_bbox(node), point);
      // If bounding box closer than previous closest entity, use gjk to
      // obtain exact distance to the convex hull of the entity
      if (r2 <= R2)
      {
        r2 = geometry::squared_distance(mesh, tree.tdim(),
                                        xtl::span(&bbox[1], 1),
                                        xt::reshape_view(point, {1, 3}))[0];
      }
    }

    // If entity is closer than best result so far, return it
    if (r2 <= R2)
    {
      closest_entity = bbox[1];
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
        tree, point, bbox[0], mesh, closest_entity, R2);
    std::pair<int, double> p1 = _compute_closest_entity(
        tree, point, bbox[1], mesh, p0.first, p0.second);
    return p1;
  }
}
//-----------------------------------------------------------------------------
/// Compute collisions with a single point
/// @param[in] tree The bounding box tree
/// @param[in] points The points (shape=(num_points, 3))
/// @param[in, out] entities The list of colliding entities (local to process)
void _compute_collisions_point(
    const geometry::BoundingBoxTree& tree,
    const xt::xtensor_fixed<double, xt::xshape<3>>& p,
    std::vector<int>& entities)
{
  std::deque<std::int32_t> stack;
  int next = tree.num_bboxes() - 1;

  while (next != -1)
  {
    std::array bbox = tree.bbox(next);
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
void _compute_collisions_tree(const geometry::BoundingBoxTree& A,
                              const geometry::BoundingBoxTree& B, int node_A,
                              int node_B,
                              std::vector<std::array<int, 2>>& entities)
{
  // If bounding boxes don't collide, then don't search further
  if (!bbox_in_bbox(A.get_bbox(node_A), B.get_bbox(node_B)))
    return;

  // Get bounding boxes for current nodes
  const std::array bbox_A = A.bbox(node_A);
  const std::array bbox_B = B.bbox(node_B);

  // Check whether we've reached a leaf in A or B
  const bool is_leaf_A = is_leaf(bbox_A);
  const bool is_leaf_B = is_leaf(bbox_B);
  if (is_leaf_A and is_leaf_B)
  {
    // If both boxes are leaves (which we know collide), then add them
    // child_1 denotes entity for leaves
    entities.push_back({bbox_A[1], bbox_B[1]});
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

} // namespace

//-----------------------------------------------------------------------------
geometry::BoundingBoxTree
geometry::create_midpoint_tree(const mesh::Mesh& mesh, int tdim,
                               const xtl::span<const std::int32_t>& entities)
{
  LOG(INFO) << "Building point search tree to accelerate distance queries for "
               "a given topological dimension and subset of entities.";

  const auto midpoints = mesh::compute_midpoints(mesh, tdim, entities);
  std::vector<std::pair<std::array<double, 3>, std::int32_t>> points(
      entities.size());
  for (std::size_t i = 0; i < points.size(); ++i)
  {
    for (std::size_t j = 0; j < 3; ++j)
      points[i].first[j] = midpoints(i, j);
    points[i].second = entities[i];
  }

  // Build tree
  return geometry::BoundingBoxTree(points);
}
//-----------------------------------------------------------------------------
std::vector<std::array<int, 2>>
geometry::compute_collisions(const BoundingBoxTree& tree0,
                             const BoundingBoxTree& tree1)
{
  // Call recursive find function
  std::vector<std::array<int, 2>> entities;
  if (tree0.num_bboxes() > 0 and tree1.num_bboxes() > 0)
  {
    _compute_collisions_tree(tree0, tree1, tree0.num_bboxes() - 1,
                             tree1.num_bboxes() - 1, entities);
  }

  return entities;
}
//-----------------------------------------------------------------------------
graph::AdjacencyList<std::int32_t>
geometry::compute_collisions(const BoundingBoxTree& tree,
                             const xt::xtensor<double, 2>& points)
{
  if (tree.num_bboxes() > 0)
  {
    std::vector<std::int32_t> entities, offsets(points.shape(0) + 1, 0);
    entities.reserve(points.shape(0));
    for (std::size_t p = 0; p < points.shape(0); ++p)
    {
      _compute_collisions_point(tree, xt::row(points, p), entities);
      offsets[p + 1] = entities.size();
    }

    return graph::AdjacencyList<std::int32_t>(std::move(entities),
                                              std::move(offsets));
  }
  else
  {
    return graph::AdjacencyList<std::int32_t>(
        std::vector<std::int32_t>(),
        std::vector<std::int32_t>(points.shape(0) + 1, 0));
  }
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t> geometry::compute_closest_entity(
    const BoundingBoxTree& tree, const BoundingBoxTree& midpoint_tree,
    const mesh::Mesh& mesh, const xt::xtensor<double, 2>& points)
{
  assert(points.shape(1) == 3);
  if (tree.num_bboxes() == 0)
    return std::vector<std::int32_t>(points.shape(0), -1);
  else
  {
    double R2;
    double initial_entity;
    std::array<int, 2> leaves;
    std::vector<std::int32_t> entities;
    entities.reserve(points.shape(0));
    for (std::size_t i = 0; i < points.shape(0); i++)
    {
      // Use midpoint tree to find initial closest entity to the point.
      // Start by using a leaf node as the initial guess for the input
      // entity
      leaves = midpoint_tree.bbox(0);
      assert(is_leaf(leaves));
      initial_entity = leaves[0];
      xt::xtensor_fixed<double, xt::xshape<3>> diff
          = xt::row(midpoint_tree.get_bbox(0), 0);
      diff -= xt::row(points, i);
      R2 = xt::norm_sq(diff)();

      // Use a recursive search through the bounding box tree
      // to find determine the entity with the closest midpoint.
      // As the midpoint tree only consist of points, the distance
      // queries are lightweight.
      const auto [m_index, m_distance2] = _compute_closest_entity(
          midpoint_tree, xt::reshape_view(xt::row(points, i), {1, 3}),
          midpoint_tree.num_bboxes() - 1, mesh, initial_entity, R2);

      // Use a recursive search through the bounding box tree to
      // determine which entity is actually closest.
      // Uses the entity with the closest midpoint as initial guess, and
      // the distance from the midpoint to the point of interest as the
      // initial search radius.
      const auto [index, distance2] = _compute_closest_entity(
          tree, xt::reshape_view(xt::row(points, i), {1, 3}),
          tree.num_bboxes() - 1, mesh, m_index, m_distance2);

      entities.push_back(index);
    }

    return entities;
  }
}

//-----------------------------------------------------------------------------
double geometry::compute_squared_distance_bbox(
    const xt::xtensor_fixed<double, xt::xshape<2, 3>>& b,
    const xt::xtensor_fixed<double, xt::xshape<3>>& x)
{
  const xt::xtensor_fixed<double, xt::xshape<3>> d0 = x - xt::row(b, 0);
  const xt::xtensor_fixed<double, xt::xshape<3>> d1 = x - xt::row(b, 1);
  auto _d0 = xt::where(d0 > 0.0, 0, d0);
  auto _d1 = xt::where(d1 < 0.0, 0, d1);
  return xt::norm_sq(_d0)() + xt::norm_sq(_d1)();
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2>
geometry::shortest_vector(const mesh::Mesh& mesh, int dim,
                          const xtl::span<const std::int32_t>& entities,
                          const xt::xtensor<double, 2>& points)
{
  assert(points.shape(1) == 3);
  const int tdim = mesh.topology().dim();
  const mesh::Geometry& geometry = mesh.geometry();
  xtl::span<const double> geom_dofs = geometry.x();
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  xt::xtensor<double, 2> shortest_vectors({entities.size(), 3});
  if (dim == tdim)
  {
    for (std::size_t e = 0; e < entities.size(); e++)
    {
      auto dofs = x_dofmap.links(entities[e]);
      xt::xtensor<double, 2> nodes({dofs.size(), 3});
      for (std::size_t i = 0; i < dofs.size(); ++i)
      {
        const int pos = 3 * dofs[i];
        for (std::size_t j = 0; j < 3; ++j)
          nodes(i, j) = geom_dofs[pos + j];
      }

      xt::row(shortest_vectors, e) = geometry::compute_distance_gjk(
          xt::reshape_view(xt::row(points, e), {1, 3}), nodes);
    }
  }
  else
  {
    mesh.topology_mutable().create_connectivity(dim, tdim);
    mesh.topology_mutable().create_connectivity(tdim, dim);
    auto e_to_c = mesh.topology().connectivity(dim, tdim);
    assert(e_to_c);
    auto c_to_e = mesh.topology_mutable().connectivity(tdim, dim);
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
      auto dofs = x_dofmap.links(c);
      const std::vector<int> entity_dofs
          = geometry.cmap().create_dof_layout().entity_closure_dofs(
              dim, local_cell_entity);
      xt::xtensor<double, 2> nodes({entity_dofs.size(), 3});
      for (std::size_t i = 0; i < entity_dofs.size(); i++)
      {
        const int pos = 3 * dofs[entity_dofs[i]];
        for (std::size_t j = 0; j < 3; ++j)
          nodes(i, j) = geom_dofs[pos + j];
      }

      xt::row(shortest_vectors, e) = compute_distance_gjk(
          xt::reshape_view(xt::row(points, e), {1, 3}), nodes);
    }
  }

  return shortest_vectors;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 1>
geometry::squared_distance(const mesh::Mesh& mesh, int dim,
                           const xtl::span<const std::int32_t>& entities,
                           const xt::xtensor<double, 2>& points)
{
  return xt::norm_sq(shortest_vector(mesh, dim, entities, points), {1});
}
//-------------------------------------------------------------------------------
graph::AdjacencyList<std::int32_t> geometry::compute_colliding_cells(
    const mesh::Mesh& mesh,
    const graph::AdjacencyList<std::int32_t>& candidate_cells,
    const xt::xtensor<double, 2>& points)
{
  std::vector<std::int32_t> offsets = {0};
  offsets.reserve(candidate_cells.num_nodes() + 1);
  std::vector<std::int32_t> colliding_cells;
  constexpr double eps2 = 1e-20;
  const int tdim = mesh.topology().dim();
  for (std::int32_t i = 0; i < candidate_cells.num_nodes(); i++)
  {
    auto cells = candidate_cells.links(i);
    xt::xtensor<double, 2> _point({cells.size(), 3});
    for (std::size_t j = 0; j < cells.size(); j++)
      xt::row(_point, j) = xt::row(points, i);

    xt::xtensor<double, 1> distances_sq
        = geometry::squared_distance(mesh, tdim, cells, _point);
    for (std::size_t j = 0; j < cells.size(); j++)
      if (distances_sq[j] < eps2)
        colliding_cells.push_back(cells[j]);

    offsets.push_back(colliding_cells.size());
  }

  return graph::AdjacencyList<std::int32_t>(std::move(colliding_cells),
                                            std::move(offsets));
}
//-------------------------------------------------------------------------------
