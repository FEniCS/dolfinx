// Copyright (C) 2006-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include "BoundingBoxTree.h"
#include "openGJK.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/log.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshEntity.h>
#include <dolfinx/mesh/utils.h>

using namespace dolfinx;

namespace
{
//-----------------------------------------------------------------------------
// Check whether bounding box is a leaf node
inline bool is_leaf(const std::array<int, 2>& bbox, int node)
{
  // Leaf nodes are marked by setting child_0 equal to the node itself
  return bbox[0] == node;
}
//-----------------------------------------------------------------------------
bool point_in_bbox(const Eigen::Array<double, 2, 3, Eigen::RowMajor>& b,
                   const Eigen::Vector3d& x)
{
  const double rtol = 1e-14;
  auto eps0 = rtol * (b.row(1) - b.row(0));
  return (x.transpose().array() >= (b.row(0) - eps0).array()).all()
         and (x.transpose().array() <= (b.row(1) + eps0).array()).all();
}
//-----------------------------------------------------------------------------
bool bbox_in_bbox(const Eigen::Array<double, 2, 3, Eigen::RowMajor>& a,
                  const Eigen::Array<double, 2, 3, Eigen::RowMajor>& b)
{
  const double rtol = 1e-14;
  auto eps0 = rtol * (b.row(1) - b.row(0));
  return (b.row(0) - eps0 <= a.row(1)).all()
         and (b.row(1) + eps0 >= a.row(0)).all();
}
//-----------------------------------------------------------------------------
// Check whether point is outside region defined by facet ABC. The
// fourth vertex is needed to define the orientation.
bool point_outside_of_plane(const Eigen::Vector3d& p, const Eigen::Vector3d& a,
                            const Eigen::Vector3d& b, const Eigen::Vector3d& c,
                            const Eigen::Vector3d& d)
{
  // Algorithm from Real-time collision detection by Christer Ericson:
  // PointOutsideOfPlane on page 144, Section 5.1.6.
  const Eigen::Vector3d v = (b - a).cross(c - a);
  const double signp = v.dot(p - a);
  const double signd = v.dot(d - a);
  return signp * signd < 0.0;
}
//-----------------------------------------------------------------------------
// Compute closest entity {closest_entity, R2} (recursive)
std::pair<int, double>
_compute_closest_entity(const geometry::BoundingBoxTree& tree,
                        const Eigen::Vector3d& point, int node,
                        const mesh::Mesh& mesh, int closest_entity, double R2)
{
  // Get children of current bounding box node
  const std::array<int, 2> bbox = tree.bbox(node);

  // If bounding box is outside radius, then don't search further
  const double r2
      = geometry::compute_squared_distance_bbox(tree.get_bbox(node), point);
  if (r2 > R2)
  {
    // If bounding box is outside radius, then don't search further
    return {closest_entity, R2};
  }
  else if (is_leaf(bbox, node))
  {
    // If box is leaf (which we know is inside radius), then shrink radius

    // Get entity (child_1 denotes entity index for leaves)
    assert(tree.tdim() == mesh.topology().dim());
    const int entity_index = bbox[1];
    mesh::MeshEntity cell(mesh, mesh.topology().dim(), entity_index);

    // If entity is closer than best result so far, then return it
    const double r2 = geometry::squared_distance(cell, point);
    if (r2 < R2)
    {
      closest_entity = entity_index;
      R2 = r2;
    }

    return {closest_entity, R2};
  }
  else
  {
    // Check both children
    std::pair<int, double> p0 = _compute_closest_entity(
        tree, point, bbox[0], mesh, closest_entity, R2);
    std::pair<int, double> p1 = _compute_closest_entity(
        tree, point, bbox[1], mesh, p0.first, p0.second);
    return p1;
  }
}
//-----------------------------------------------------------------------------
// Compute closest point {closest_point, R2} (recursive)
std::pair<int, double>
_compute_closest_point(const geometry::BoundingBoxTree& tree,
                       const Eigen::Vector3d& point, int node,
                       int closest_point, double R2)
{
  // Get children of current bounding box node
  const std::array<int, 2> bbox = tree.bbox(node);

  // If box is leaf, then compute distance and shrink radius
  if (is_leaf(bbox, node))
  {
    const double r2 = (tree.get_bbox(node).row(0).transpose().matrix() - point)
                          .squaredNorm();

    if (r2 < R2)
    {
      closest_point = bbox[1];
      R2 = r2;
    }

    return {closest_point, R2};
  }
  else
  {
    // If bounding box is outside radius, then don't search further
    const double r2
        = geometry::compute_squared_distance_bbox(tree.get_bbox(node), point);
    if (r2 > R2)
      return {closest_point, R2};

    // Check both children
    std::pair<int, double> p0
        = _compute_closest_point(tree, point, bbox[0], closest_point, R2);
    std::pair<int, double> p1
        = _compute_closest_point(tree, point, bbox[1], p0.first, p0.second);
    return p1;
  }
}
//-----------------------------------------------------------------------------
// Compute collisions with point (recursive)
void _compute_collisions_point(const geometry::BoundingBoxTree& tree,
                               const Eigen::Vector3d& p, int node,
                               std::vector<int>& entities)
{
  // Get children of current bounding box node
  const std::array<int, 2> bbox = tree.bbox(node);

  if (!point_in_bbox(tree.get_bbox(node), p))
  {
    // If point is not in bounding box, then don't search further
    return;
  }
  else if (is_leaf(bbox, node))
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
  const std::array<int, 2> bbox_A = A.bbox(node_A);
  const std::array<int, 2> bbox_B = B.bbox(node_B);

  // Check whether we've reached a leaf in A or B
  const bool is_leaf_A = is_leaf(bbox_A, node_A);
  const bool is_leaf_B = is_leaf(bbox_B, node_B);
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
geometry::BoundingBoxTree geometry::create_midpoint_tree(const mesh::Mesh& mesh)
{
  LOG(INFO) << "Building point search tree to accelerate distance queries.";

  // Create list of midpoints for all cells
  const int dim = mesh.topology().dim();
  auto map = mesh.topology().index_map(dim);
  assert(map);
  const std::int32_t num_cells = map->size_local() + map->num_ghosts();
  Eigen::Array<int, Eigen::Dynamic, 1> entities(num_cells);
  std::iota(entities.data(), entities.data() + entities.rows(), 0);
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> midpoints
      = mesh::midpoints(mesh, dim, entities);

  std::vector<Eigen::Vector3d> points(entities.rows());
  for (std::size_t i = 0; i < points.size(); ++i)
    points[i] = midpoints.row(i);

  // Build tree
  return geometry::BoundingBoxTree(points);
}
//-----------------------------------------------------------------------------
std::vector<std::array<int, 2>>
geometry::compute_collisions(const BoundingBoxTree& tree0,
                             const BoundingBoxTree& tree1)
{
  std::vector<std::array<int, 2>> entities;

  // Call recursive find function
  _compute_collisions_tree(tree0, tree1, tree0.num_bboxes() - 1,
                           tree1.num_bboxes() - 1, entities);

  return entities;
}
//-----------------------------------------------------------------------------
std::vector<int> geometry::compute_collisions(const BoundingBoxTree& tree,
                                              const Eigen::Vector3d& p)
{
  std::vector<int> entities;
  _compute_collisions_point(tree, p, tree.num_bboxes() - 1, entities);
  return entities;
}
//-----------------------------------------------------------------------------
std::vector<int>
geometry::compute_process_collisions(const geometry::BoundingBoxTree& tree,
                                     const Eigen::Vector3d& p)
{
  if (tree.global_tree)
    return geometry::compute_collisions(*tree.global_tree, p);
  else
  {
    std::vector<int> collision;
    if (point_in_bbox(tree.get_bbox(tree.num_bboxes() - 1), p))
      collision.push_back(0);
    return collision;
  }
}
//-----------------------------------------------------------------------------
double geometry::compute_squared_distance_bbox(
    const Eigen::Array<double, 2, 3, Eigen::RowMajor>& b,
    const Eigen::Vector3d& x)
{
  auto d0 = x.array() - b.row(0).transpose();
  auto d1 = x.array() - b.row(1).transpose();
  return (d0 > 0.0).select(0, d0).matrix().squaredNorm()
         + (d1 < 0.0).select(0, d1).matrix().squaredNorm();
}
//-----------------------------------------------------------------------------
std::pair<int, double> geometry::compute_closest_entity(
    const BoundingBoxTree& tree, const BoundingBoxTree& tree_midpoint,
    const Eigen::Vector3d& p, const mesh::Mesh& mesh)
{
  // Closest entity only implemented for cells. Consider extending this.
  if (tree.tdim() != mesh.topology().dim())
  {
    throw std::runtime_error("Cannot compute closest entity of point. "
                             "Closest-entity is only implemented for cells");
  }

  // Search point cloud to get a good starting guess
  std::pair<int, double> guess = compute_closest_point(tree_midpoint, p);
  const double r = guess.second;

  // Return if we have found the point
  if (r == 0.0)
    return guess;

  // Call recursive find function
  std::pair<int, double> e = _compute_closest_entity(
      tree, p, tree.num_bboxes() - 1, mesh, guess.first, r * r);

  // Sanity check
  assert(e.first >= 0);

  e.second = sqrt(e.second);
  return e;
}
//-----------------------------------------------------------------------------
std::pair<int, double>
geometry::compute_closest_point(const BoundingBoxTree& tree,
                                const Eigen::Vector3d& p)
{
  // Closest point only implemented for point cloud
  if (tree.tdim() != 0)
  {
    throw std::runtime_error("Cannot compute closest point. "
                             "Search tree has not been built for point cloud");
  }

  // Note that we don't compute a point search tree here... That would
  // be weird.

  // Get initial guess by picking the distance to a "random" point
  int closest_point = 0;
  // double R2 = tree.compute_squared_distance_point(p, closest_point);
  const double R2
      = (tree.get_bbox(closest_point).row(0).transpose().matrix() - p)
            .squaredNorm();

  // Call recursive find function
  _compute_closest_point(tree, p, tree.num_bboxes() - 1, closest_point, R2);

  return {closest_point, sqrt(R2)};
}
//-----------------------------------------------------------------------------
double geometry::squared_distance(const mesh::MeshEntity& entity,
                                  const Eigen::Vector3d& p)
{
  const int dim = entity.dim();
  const int tdim = entity.mesh().topology().dim();
  const graph::AdjacencyList<std::int32_t>& x_dofmap
      = entity.mesh().geometry().dofmap();

  // Find attached cell
  entity.mesh().topology_mutable().create_connectivity(dim, tdim);
  auto e_to_c = entity.mesh().topology().connectivity(dim, tdim);
  assert(e_to_c);
  assert(e_to_c->num_links(entity.index()) > 0);
  const std::int32_t c = e_to_c->links(entity.index())[0];

  auto dofs = x_dofmap.links(c);
  auto c_to_v = entity.mesh().topology().connectivity(tdim, 0);
  assert(c_to_v);
  auto cell_vertices = c_to_v->links(c);

  auto vertices = entity.entities(0);
  const mesh::CellType type = entity.mesh().topology().cell_type();
  const mesh::Geometry& geometry = entity.mesh().geometry();

  Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> v(vertices.size(),
                                                              3);
  for (int i = 0; i < vertices.size(); ++i)
  {
    const std::int32_t* it
        = std::find(cell_vertices.data(),
                    cell_vertices.data() + cell_vertices.rows(), vertices[i]);
    assert(it != (cell_vertices.data() + cell_vertices.rows()));
    const int local_vertex = std::distance(cell_vertices.data(), it);
    v.row(i) = geometry.node(dofs(local_vertex));
  }

  return geometry::gjk_vector(p.transpose(), v).squaredNorm();
}
//-------------------------------------------------------------------------------
std::vector<int>
geometry::select_cells_from_candidates(const dolfinx::mesh::Mesh& mesh,
                                       const std::vector<int>& candidate_cells,
                                       const Eigen::Vector3d& point, int n)
{
  const double eps2 = 1e-20;
  const int tdim = mesh.topology().dim();
  std::vector<int> result;
  for (int c : candidate_cells)
  {
    mesh::MeshEntity entity(mesh, tdim, c);
    const double d2 = squared_distance(entity, point);
    if (d2 < eps2)
    {
      result.push_back(c);
      if (result.size() == n)
        return result;
    }
  }
  return result;
}
