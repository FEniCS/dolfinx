// Copyright (C) 2006-2021 Chris N. Richardson, Anders Logg, Garth N. Wells and
// Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include "BoundingBoxTree.h"
#include "gjk.h"
#include <dolfinx/common/log.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>
#include <xtensor/xfixed.hpp>
#include <xtensor/xnorm.hpp>

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
bool point_in_bbox(const xt::xtensor_fixed<double, xt::xshape<2, 3>>& b,
                   const xt::xtensor_fixed<double, xt::xshape<3>>& x)
{
  auto b0 = xt::row(b, 0);
  auto b1 = xt::row(b, 1);
  constexpr double rtol = 1e-14;
  auto eps0 = rtol * (b1 - b0);
  return xt::all(x >= (b0 - eps0)) and xt::all(x <= (b1 + eps0));
}
//-----------------------------------------------------------------------------
bool bbox_in_bbox(const xt::xtensor_fixed<double, xt::xshape<2, 3>>& a,
                  const xt::xtensor_fixed<double, xt::xshape<2, 3>>& b)
{
  constexpr double rtol = 1e-14;
  auto b0 = xt::row(b, 0);
  auto b1 = xt::row(b, 1);
  auto a0 = xt::row(a, 0);
  auto a1 = xt::row(a, 1);
  auto eps0 = rtol * (b1 - b0);
  return xt::all(b0 - eps0 <= a1) and xt::all(b1 + eps0 >= a0);
}
//-----------------------------------------------------------------------------
// Compute closest entity {closest_entity, R2} (recursive)
std::pair<std::int32_t, double> _compute_closest_entity(
    const geometry::BoundingBoxTree& tree,
    const xt::xtensor_fixed<double, xt::xshape<1, 3>>& point, int node,
    const mesh::Mesh& mesh, std::int32_t closest_entity, double R2)
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
      diff -= xt::row(point, 0);
      r2 = xt::norm_sq(diff)();
    }
    else
    {
      r2 = geometry::compute_squared_distance_bbox(tree.get_bbox(node), point);
      // If bounding box closer than previous closest entity, use gjk to
      // obtain exact distance to the convex hull of the entity
      if (r2 <= R2)
      {
        const std::array<std::int32_t, 1> index = {bbox[1]};
        r2 = geometry::squared_distance(mesh, tree.tdim(), index, point)[0];
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
// Compute collisions with point (recursive)
void _compute_collisions_point(
    const geometry::BoundingBoxTree& tree,
    const xt::xtensor_fixed<double, xt::xshape<3>>& p, int node,
    std::vector<int>& entities)
{
  // Get children of current bounding box node
  const std::array bbox = tree.bbox(node);

  if (!point_in_bbox(tree.get_bbox(node), p))
  {
    // If point is not in bounding box, then don't search further
    return;
  }
  else if (is_leaf(bbox))
  {
    // If box is a leaf (which we know contains the point), then add it

    // child_1 denotes entity for leaves
    const int entity_index = bbox[1];

    // Add the candidate
    entities.push_back(entity_index);
  }
  else
  {
    // Check both children
    _compute_collisions_point(tree, p, bbox[0], entities);
    _compute_collisions_point(tree, p, bbox[1], entities);
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
    // tree first. Note that nodes are added in reverse order with the top
    // bounding box at the end so the largest tree (the one with the the
    // most boxes left to traverse) has the largest node number.
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

  const auto midpoints = mesh::midpoints(mesh, tdim, entities);
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
dolfinx::graph::AdjacencyList<int>
geometry::compute_collisions(const BoundingBoxTree& tree,
                             const xt::xtensor<double, 2>& points)
{
  const std::size_t num_points = points.shape(0);
  std::vector<int> entities;
  std::vector<std::int32_t> offsets({0});
  offsets.reserve(num_points);
  entities.reserve(num_points);
  for (std::size_t i = 0; i < num_points; i++)
  {
    if (tree.num_bboxes() > 0)
    {
      _compute_collisions_point(tree, xt::row(points, i), tree.num_bboxes() - 1,
                                entities);
      offsets.push_back(entities.size());
    }
  }

  return dolfinx::graph::AdjacencyList<int>(entities, offsets);
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
std::vector<std::int32_t> geometry::compute_closest_entity(
    const BoundingBoxTree& tree, const BoundingBoxTree& midpoint_tree,
    xt::xtensor<double, 2>& points, const mesh::Mesh& mesh)
{
  assert(points.shape(1) == 3);

  std::vector<std::int32_t> entities;
  entities.reserve(points.shape(0));
  const std::size_t num_points = points.shape(0);
  if (tree.num_bboxes() == 0)
  {
    for (std::size_t i = 0; i < num_points; i++)
      entities.push_back(-1);
    return entities;
  }
  else
  {
    double R2;
    const double initial_entity = 0;
    xt::xtensor_fixed<double, xt::xshape<1, 3>> _point;
    for (std::size_t i = 0; i < num_points; i++)
    {
      // Use midpoint tree to find intial closest entity to the point
      // Start by using a leaf node as the initial guess for the input entity
      xt::xtensor_fixed<double, xt::xshape<3>> diff
          = xt::row(midpoint_tree.get_bbox(initial_entity), 0);
      diff -= xt::row(points, i);
      R2 = xt::norm_sq(diff)();

      // Use a recursive search through the bounding box tree
      // to find determine the entity with the closest midpoint.
      // As the midpoint tree only consist of points, the distance queries are
      // lightweight.
      _point = xt::row(points, i);
      const auto [m_index, m_distance2] = _compute_closest_entity(
          midpoint_tree, _point, midpoint_tree.num_bboxes() - 1, mesh, 0, R2);

      // Use a recursive search through the bounding box tree to determine which
      // entity is actually closest.
      // Uses the entity with the closest midpoint as initial guess, and the
      // distance from the midpoint to the point of interest as the initial
      // search radius.
      const auto [index, distance2] = _compute_closest_entity(
          tree, _point, tree.num_bboxes() - 1, mesh, m_index, m_distance2);

      entities.push_back(index);
    }

    return entities;
  }
}

//-----------------------------------------------------------------------------
xt::xtensor<double, 2>
geometry::distance(const mesh::Mesh& mesh, int dim,
                   const xtl::span<const std::int32_t>& entities,
                   const xt::xtensor<double, 2>& points)
{
  assert(points.shape(1) == 3);
  const int tdim = mesh.topology().dim();
  const mesh::Geometry& geometry = mesh.geometry();
  const xt::xtensor<double, 2>& geom_dofs = geometry.x();
  assert(geom_dofs.shape(1) == 3);

  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();

  xt::xtensor<double, 2> distances(
      {entities.size(), static_cast<std::size_t>(3)});
  xt::xtensor_fixed<double, xt::xshape<1, 3>> _point;
  if (dim == tdim)
  {
    for (std::size_t e = 0; e < entities.size(); e++)
    {
      auto dofs = x_dofmap.links(entities[e]);
      xt::xtensor<double, 2> nodes({dofs.size(), 3});
      for (std::size_t i = 0; i < dofs.size(); ++i)
        for (std::size_t j = 0; j < 3; ++j)
          nodes(i, j) = geom_dofs(dofs[i], j);
      _point = xt::row(points, e);
      xt::row(distances, e) = geometry::compute_distance_gjk(_point, nodes);
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
          = geometry.cmap().dof_layout().entity_closure_dofs(dim,
                                                             local_cell_entity);
      xt::xtensor<double, 2> nodes({entity_dofs.size(), 3});
      for (std::size_t i = 0; i < entity_dofs.size(); i++)
        for (std::size_t j = 0; j < 3; ++j)
          nodes(i, j) = geom_dofs(dofs[entity_dofs[i]], j);
      _point = xt::row(points, e);
      xt::row(distances, e) = geometry::compute_distance_gjk(_point, nodes);
    }
  }
  return distances;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 1>
geometry::squared_distance(const mesh::Mesh& mesh, int dim,
                           const xtl::span<const std::int32_t>& entities,
                           const xt::xtensor<double, 2>& points)
{
  return xt::norm_sq(geometry::distance(mesh, dim, entities, points), {1});
}
//-------------------------------------------------------------------------------
dolfinx::graph::AdjacencyList<int> geometry::select_colliding_cells(
    const mesh::Mesh& mesh,
    const dolfinx::graph::AdjacencyList<int>& candidate_cells,
    const xt::xtensor<double, 2>& points)
{

  const std::int32_t num_nodes = candidate_cells.num_nodes();
  std::vector<std::int32_t> offsets = {0};
  offsets.reserve(num_nodes + 1);
  std::vector<std::int32_t> colliding_cells;
  colliding_cells.reserve(candidate_cells.offsets().back());

  const double eps2 = 1e-20;
  const int tdim = mesh.topology().dim();
  std::vector<std::int32_t> result;
  for (std::int32_t i = 0; i < num_nodes; i++)
  {

    auto cells = candidate_cells.links(i);
    std::size_t num_cells = candidate_cells.num_links(i);
    xt::xtensor<double, 2> _point({num_cells, static_cast<std::size_t>(3)});
    for (std::int32_t j = 0; j < num_cells; j++)
      xt::row(_point, j) = xt::row(points, i);
    xt::xtensor<double, 1> distances_sq
        = geometry::squared_distance(mesh, tdim, cells, _point);
    for (int j = 0; j < cells.size(); j++)
      if (distances_sq[j] < eps2)
        colliding_cells.push_back(cells[j]);
    offsets.push_back(colliding_cells.size());
  }
  return dolfinx::graph::AdjacencyList<int>(colliding_cells, offsets);
}
//-------------------------------------------------------------------------------
