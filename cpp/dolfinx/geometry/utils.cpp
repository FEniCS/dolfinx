// Copyright (C) 2006-2021 Chris N. Richardson, Anders Logg, Garth N. Wells and
// Jørgen S. Dokken
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include "BoundingBoxTree.h"
#include "gjk.h"
#include <dolfinx/common/log.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>
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
bool point_in_bbox(const std::array<std::array<double, 3>, 2>& b,
                   const std::array<double, 3>& x)
{
  Eigen::Array3d _x, b0, b1;
  for (int i = 0; i < 3; ++i)
  {
    _x[i] = x[i];
    b0[i] = b[0][i];
    b1[i] = b[1][i];
  }

  const double rtol = 1e-14;
  const Eigen::Array3d eps0 = rtol * (b1 - b0);
  return (_x >= (b0 - eps0)).all() and (_x <= (b1 + eps0)).all();
}
//-----------------------------------------------------------------------------
bool bbox_in_bbox(const std::array<std::array<double, 3>, 2>& a,
                  const std::array<std::array<double, 3>, 2>& b)
{
  Eigen::Array3d a0, a1, b0, b1;
  for (int i = 0; i < 3; ++i)
  {
    a0[i] = a[0][i];
    a1[i] = a[1][i];
    b0[i] = b[0][i];
    b1[i] = b[1][i];
  }

  constexpr double rtol = 1e-14;
  auto eps0 = rtol * (b1 - b0);
  return (b0 - eps0 <= a1).all() and (b1 + eps0 >= a0).all();
}
//-----------------------------------------------------------------------------
// Compute closest entity {closest_entity, R2} (recursive)
std::pair<std::int32_t, double> _compute_closest_entity(
    const geometry::BoundingBoxTree& tree, const std::array<double, 3>& point,
    int node, const mesh::Mesh& mesh, std::int32_t closest_entity, double R2)
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
      const std::array<double, 3> x = tree.get_bbox(node)[0];
      r2 = (x[0] - point[0]) * (x[0] - point[0])
           + (x[1] - point[1]) * (x[1] - point[1])
           + (x[2] - point[2]) * (x[2] - point[2]);
    }
    else
    {
      r2 = geometry::compute_squared_distance_bbox(tree.get_bbox(node), point);
      // If bounding box closer than previous closest entity, use gjk to
      // obtain exact distance to the convex hull of the entity
      if (r2 <= R2)
      {
        r2 = geometry::squared_distance(mesh, tree.tdim(), bbox[1], point);
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
void _compute_collisions_point(const geometry::BoundingBoxTree& tree,
                               const std::array<double, 3>& p, int node,
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
                               const std::vector<std::int32_t>& entities)
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
std::vector<int> geometry::compute_collisions(const BoundingBoxTree& tree,
                                              const std::array<double, 3>& p)
{
  std::vector<int> entities;
  if (tree.num_bboxes() > 0)
    _compute_collisions_point(tree, p, tree.num_bboxes() - 1, entities);

  return entities;
}
//-----------------------------------------------------------------------------
double geometry::compute_squared_distance_bbox(
    const std::array<std::array<double, 3>, 2>& b,
    const std::array<double, 3>& x)
{
  Eigen::Array3d d0, d1;
  for (int i = 0; i < 3; ++i)
  {
    d0[i] = x[i] - b[0][i];
    d1[i] = x[i] - b[1][i];
  }

  return (d0 > 0.0).select(0, d0).matrix().squaredNorm()
         + (d1 < 0.0).select(0, d1).matrix().squaredNorm();
}
//-----------------------------------------------------------------------------
std::pair<int, double>
geometry::compute_closest_entity(const BoundingBoxTree& tree,
                                 const std::array<double, 3>& p,
                                 const mesh::Mesh& mesh, double R)
{
  // If bounding box tree is empty (on this processor) end search
  if (tree.num_bboxes() == 0)
    return {-1, -1};

  // If initial search radius is 0 we estimate the initial distance to
  // the point using the first node in the tree
  double R2 = 0.0;
  std::int32_t initial_guess;
  if (R < 0.0)
  {
    const std::array<double, 3> x = tree.get_bbox(0)[0];
    R2 = (x[0] - p[0]) * (x[0] - p[0]) + (x[1] - p[1]) * (x[1] - p[1])
         + (x[2] - p[2]) * (x[2] - p[2]);
    initial_guess = 0;
  }
  else
  {
    R2 = R * R;
    initial_guess = -1;
  }

  // Use GJK to find determine the actual closest entity
  const auto [index, distance2] = _compute_closest_entity(
      tree, p, tree.num_bboxes() - 1, mesh, initial_guess, R2);
  if (index < 0)
  {
    throw std::runtime_error("No entity found within radius "
                             + std::to_string(std::sqrt(R2)) + ".");
  }

  return {index, std::sqrt(distance2)};
}
//-----------------------------------------------------------------------------
double geometry::squared_distance(const mesh::Mesh& mesh, int dim,
                                  std::int32_t index,
                                  const std::array<double, 3>& p)
{
  const int tdim = mesh.topology().dim();
  const mesh::Geometry& geometry = mesh.geometry();
  // FIXME: Use eigen map for now.
  Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>>
      geom_dofs(geometry.x().data(), geometry.x().shape[0], geometry.x().shape[1]);

  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();

  Eigen::Vector3d _p;
  _p << p[0], p[1], p[2];

  if (dim == tdim)
  {
    auto dofs = x_dofmap.links(index);
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> nodes(dofs.size(),
                                                                    3);
    for (std::size_t i = 0; i < dofs.size(); i++)
      nodes.row(i) = geom_dofs.row(dofs[i]);

    const std::array<double, 3> x
        = geometry::compute_distance_gjk(_p.transpose(), nodes);
    return x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
  }
  else
  {
    // Find attached cell
    mesh.topology_mutable().create_connectivity(dim, tdim);
    auto e_to_c = mesh.topology().connectivity(dim, tdim);
    assert(e_to_c);
    assert(e_to_c->num_links(index) > 0);
    const std::int32_t c = e_to_c->links(index)[0];

    // Find local number of entity wrt cell
    mesh.topology_mutable().create_connectivity(tdim, dim);
    auto c_to_e = mesh.topology_mutable().connectivity(tdim, dim);
    assert(c_to_e);
    auto cell_entities = c_to_e->links(c);
    auto it0 = std::find(cell_entities.begin(), cell_entities.end(), index);
    assert(it0 != cell_entities.end());
    const int local_cell_entity = std::distance(cell_entities.begin(), it0);

    // Tabulate geometry dofs for the entity
    auto dofs = x_dofmap.links(c);
    const std::vector<int> entity_dofs
        = geometry.cmap().dof_layout().entity_closure_dofs(dim,
                                                           local_cell_entity);
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> nodes(
        entity_dofs.size(), 3);
    for (std::size_t i = 0; i < entity_dofs.size(); i++)
      nodes.row(i) = geom_dofs.row(dofs[entity_dofs[i]]);

    std::array<double, 3> x
        = geometry::compute_distance_gjk(_p.transpose(), nodes);
    return x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
  }
}
//-------------------------------------------------------------------------------
std::vector<std::int32_t> geometry::select_colliding_cells(
    const mesh::Mesh& mesh,
    const tcb::span<const std::int32_t>& candidate_cells,
    const std::array<double, 3>& p, int n)
{
  const double eps2 = 1e-20;
  const int tdim = mesh.topology().dim();
  std::vector<std::int32_t> result;
  for (std::int32_t c : candidate_cells)
  {
    const double d2 = squared_distance(mesh, tdim, c, p);
    if (d2 < eps2)
    {
      result.push_back(c);
      if ((int)result.size() == n)
        return result;
    }
  }
  return result;
}
//-------------------------------------------------------------------------------
