// Copyright (C) 2013 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

// Define a maximum dimension used for a local array in the recursive
// build function. Speeds things up compared to allocating it in each
// recursion and is more convenient than sending it around.
#define MAX_DIM 6

#include "BoundingBoxTree.h"
#include "CollisionPredicates.h"
#include "utils.h"
#include <dolfin/common/MPI.h>
#include <dolfin/common/log.h>
#include <dolfin/mesh/Geometry.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/utils.h>

using namespace dolfin;
using namespace dolfin::geometry;

namespace
{
// Check whether bounding box is a leaf node
bool is_leaf(const BoundingBoxTree::BBox& bbox, int node)
{
  // Leaf nodes are marked by setting child_0 equal to the node itself
  return bbox[0] == node;
}

} // namespace

//-----------------------------------------------------------------------------
BoundingBoxTree::BoundingBoxTree(const std::vector<double>& leaf_bboxes,
                                 const std::vector<int>::iterator& begin,
                                 const std::vector<int>::iterator& end,
                                 int gdim)
    : _tdim(0), _gdim(gdim)
{
  _build_from_leaf(leaf_bboxes, begin, end);
}
//-----------------------------------------------------------------------------
BoundingBoxTree::BoundingBoxTree(const mesh::Mesh& mesh, int tdim)
    : _tdim(tdim), _gdim(mesh.topology().dim())
{
  // Check dimension
  if (tdim < 1 or tdim > mesh.topology().dim())
  {
    throw std::runtime_error("Dimension must be a number between 1 and "
                             + std::to_string(mesh.topology().dim()));
  }

  // Store topological dimension (only used for checking that entity
  // collisions can only be computed with cells)
  _tdim = tdim;

  // Initialize entities of given dimension if they don't exist
  mesh.create_entities(tdim);

  // Create bounding boxes for all entities (leaves)
  const int num_leaves = mesh.num_entities(tdim);
  std::vector<double> leaf_bboxes(2 * _gdim * num_leaves);
  for (auto& it : mesh::MeshRange(mesh, tdim))
  {
    compute_bbox_of_entity(leaf_bboxes.data() + 2 * _gdim * it.index(), it,
                           _gdim);
  }

  // Create leaf partition (to be sorted)
  std::vector<int> leaf_partition(num_leaves);
  std::iota(leaf_partition.begin(), leaf_partition.end(), 0);

  // Recursively build the bounding box tree from the leaves
  _build_from_leaf(leaf_bboxes, leaf_partition.begin(), leaf_partition.end());

  LOG(INFO) << "Computed bounding box tree with " << num_bboxes()
            << " nodes for " << num_leaves << " entities.";

  // Build tree for each process
  const std::size_t mpi_size = MPI::size(mesh.mpi_comm());
  if (mpi_size > 1)
  {
    // Send root node coordinates to all processes
    std::vector<double> send_bbox(_bbox_coordinates.end() - _gdim * 2,
                                  _bbox_coordinates.end());
    std::vector<double> recv_bbox;
    MPI::all_gather(mesh.mpi_comm(), send_bbox, recv_bbox);
    std::vector<int> global_leaves(mpi_size);
    std::iota(global_leaves.begin(), global_leaves.end(), 0);
    _global_tree.reset(new BoundingBoxTree(recv_bbox, global_leaves.begin(),
                                           global_leaves.end(), _gdim));
    LOG(INFO) << "Computed global bounding box tree with "
              << _global_tree->num_bboxes() << " boxes.";
  }
}
//-----------------------------------------------------------------------------
BoundingBoxTree::BoundingBoxTree(const std::vector<Eigen::Vector3d>& points,
                                 int gdim)
    : _tdim(0), _gdim(gdim)
{
  // Create leaf partition (to be sorted)
  const int num_leaves = points.size();
  std::vector<int> leaf_partition(num_leaves);
  std::iota(leaf_partition.begin(), leaf_partition.end(), 0);

  // Recursively build the bounding box tree from the leaves
  _build_from_point(points, leaf_partition.begin(), leaf_partition.end());

  LOG(INFO) << "Computed bounding box tree with " << num_bboxes()
            << " nodes for " << num_leaves << " points.";
}
//-----------------------------------------------------------------------------
std::vector<int>
BoundingBoxTree::compute_collisions(const Eigen::Vector3d& point) const
{
  // Call recursive find function
  std::vector<int> entities;
  _compute_collisions_point(*this, point, num_bboxes() - 1, entities, nullptr);

  return entities;
}
//-----------------------------------------------------------------------------
std::pair<std::vector<int>, std::vector<int>>
BoundingBoxTree::compute_collisions(const BoundingBoxTree& tree) const
{
  // Introduce new variables for clarity
  const BoundingBoxTree& A(*this);
  const BoundingBoxTree& B(tree);

  // Create data structures for storing collisions
  std::vector<int> entities_A;
  std::vector<int> entities_B;

  // Call recursive find function
  _compute_collisions_tree(A, B, A.num_bboxes() - 1, B.num_bboxes() - 1,
                           entities_A, entities_B, 0, 0);

  return std::make_pair(entities_A, entities_B);
}
//-----------------------------------------------------------------------------
std::vector<int>
BoundingBoxTree::compute_entity_collisions(const Eigen::Vector3d& point,
                                           const mesh::Mesh& mesh) const
{
  // Point in entity only implemented for cells. Consider extending this.
  if (_tdim != mesh.topology().dim())
  {
    throw std::runtime_error(
        "Cannot compute collision between point and mesh entities. "
        "Point-in-entity is only implemented for cells");
  }

  // Call recursive find function to compute bounding box candidates
  std::vector<int> entities;
  _compute_collisions_point(*this, point, num_bboxes() - 1, entities, &mesh);

  return entities;
}
//-----------------------------------------------------------------------------
std::vector<int>
BoundingBoxTree::compute_process_collisions(const Eigen::Vector3d& point) const
{
  if (_global_tree)
    return _global_tree->compute_collisions(point);

  std::vector<int> collision;
  if (point_in_bbox(point.data(), num_bboxes() - 1))
    collision.push_back(0);

  return collision;
}
//-----------------------------------------------------------------------------
std::pair<std::vector<int>, std::vector<int>>
BoundingBoxTree::compute_entity_collisions(const BoundingBoxTree& tree,
                                           const mesh::Mesh& mesh_A,
                                           const mesh::Mesh& mesh_B) const
{
  // Introduce new variables for clarity
  const BoundingBoxTree& A(*this);
  const BoundingBoxTree& B(tree);

  // Create data structures for storing collisions
  std::vector<int> entities_A;
  std::vector<int> entities_B;

  // Call recursive find function
  _compute_collisions_tree(A, B, A.num_bboxes() - 1, B.num_bboxes() - 1,
                           entities_A, entities_B, &mesh_A, &mesh_B);

  return std::make_pair(entities_A, entities_B);
}
//-----------------------------------------------------------------------------
int BoundingBoxTree::compute_first_collision(const Eigen::Vector3d& point) const
{
  // Call recursive find function
  return _compute_first_collision(*this, point, num_bboxes() - 1);
}
//-----------------------------------------------------------------------------
int BoundingBoxTree::compute_first_entity_collision(
    const Eigen::Vector3d& point, const mesh::Mesh& mesh) const
{
  // Point in entity only implemented for cells. Consider extending this.
  if (_tdim != mesh.topology().dim())
  {
    throw std::runtime_error(
        "Cannot compute collision between point and mesh entities. "
        "Point-in-entity is only implemented for cells");
  }

  // Call recursive find function
  return _compute_first_entity_collision(*this, point, num_bboxes() - 1, mesh);
}
//-----------------------------------------------------------------------------
std::pair<int, double>
BoundingBoxTree::compute_closest_entity(const Eigen::Vector3d& point,
                                        const mesh::Mesh& mesh) const
{
  // Closest entity only implemented for cells. Consider extending this.
  if (_tdim != mesh.topology().dim())
  {
    throw std::runtime_error("Cannot compute closest entity of point. "
                             "Closest-entity is only implemented for cells");
  }

  // Compute point search tree if not already done
  build_point_search_tree(mesh);

  // Search point cloud to get a good starting guess
  assert(_point_search_tree);
  std::pair<int, double> guess
      = _point_search_tree->compute_closest_point(point);
  double r = guess.second;

  // Return if we have found the point
  if (r == 0.)
    return guess;

  // Initialize index and distance to closest entity
  int closest_entity = -1;
  double R2 = r * r;

  // Call recursive find function
  _compute_closest_entity(*this, point, num_bboxes() - 1, mesh, closest_entity,
                          R2);

  // Sanity check
  assert(closest_entity >= 0);

  return {closest_entity, sqrt(R2)};
}
//-----------------------------------------------------------------------------
std::pair<int, double>
BoundingBoxTree::compute_closest_point(const Eigen::Vector3d& point) const
{
  // Closest point only implemented for point cloud
  if (_tdim != 0)
  {
    throw std::runtime_error("Cannot compute closest point. "
                             "Search tree has not been built for point cloud");
  }

  // Note that we don't compute a point search tree here... That would
  // be weird.

  // Get initial guess by picking the distance to a "random" point
  int closest_point = 0;
  double R2 = compute_squared_distance_point(point.data(), closest_point);

  // Call recursive find function
  _compute_closest_point(*this, point, num_bboxes() - 1, closest_point, R2);

  return {closest_point, sqrt(R2)};
}
//-----------------------------------------------------------------------------
// Implementation of private functions
//-----------------------------------------------------------------------------
int BoundingBoxTree::_build_from_leaf(const std::vector<double>& leaf_bboxes,
                                      const std::vector<int>::iterator& begin,
                                      const std::vector<int>::iterator& end)
{
  assert(begin < end);

  // Create empty bounding box data
  BBox bbox;

  // Reached leaf
  if (end - begin == 1)
  {
    // Get bounding box coordinates for leaf
    const int entity_index = *begin;
    const double* b = leaf_bboxes.data() + 2 * _gdim * entity_index;

    // Store bounding box data
    bbox[0] = num_bboxes(); // child_0 == node denotes a leaf
    bbox[1] = entity_index; // index of entity contained in leaf
    return add_bbox(bbox, b);
  }

  // Compute bounding box of all bounding boxes
  double b[MAX_DIM];
  std::size_t axis;
  compute_bbox_of_bboxes(b, axis, leaf_bboxes, begin, end, _gdim);

  // Sort bounding boxes along longest axis
  std::vector<int>::iterator middle = begin + (end - begin) / 2;
  sort_bboxes(axis, leaf_bboxes, begin, middle, end, _gdim);

  // Split bounding boxes into two groups and call recursively
  bbox[0] = _build_from_leaf(leaf_bboxes, begin, middle);
  bbox[1] = _build_from_leaf(leaf_bboxes, middle, end);

  // Store bounding box data. Note that root box will be added last.
  return add_bbox(bbox, b);
}
//-----------------------------------------------------------------------------
int BoundingBoxTree::_build_from_point(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<int>::iterator& begin,
    const std::vector<int>::iterator& end)
{
  assert(begin < end);

  // Create empty bounding box data
  BBox bbox;

  // Reached leaf
  if (end - begin == 1)
  {
    // Store bounding box data
    const int point_index = *begin;
    bbox[0] = num_bboxes(); // child_0 == node denotes a leaf
    bbox[1] = point_index;  // index of entity contained in leaf
    return add_point(bbox, points[point_index]);
  }

  // Compute bounding box of all points
  double b[MAX_DIM];
  std::size_t axis;
  compute_bbox_of_points(b, axis, points, begin, end, _gdim);

  // Sort bounding boxes along longest axis
  std::vector<int>::iterator middle = begin + (end - begin) / 2;
  sort_points(axis, points, begin, middle, end);

  // Split bounding boxes into two groups and call recursively
  bbox[0] = _build_from_point(points, begin, middle);
  bbox[1] = _build_from_point(points, middle, end);

  // Store bounding box data. Note that root box will be added last.
  return add_bbox(bbox, b);
}
//-----------------------------------------------------------------------------
void BoundingBoxTree::_compute_collisions_point(const BoundingBoxTree& tree,
                                                const Eigen::Vector3d& point,
                                                int node,
                                                std::vector<int>& entities,
                                                const mesh::Mesh* mesh)
{
  // Get bounding box for current node
  const BBox& bbox = tree._bboxes[node];

  // If point is not in bounding box, then don't search further
  if (!tree.point_in_bbox(point.data(), node))
    return;

  // If box is a leaf (which we know contains the point), then add it
  else if (is_leaf(bbox, node))
  {
    // child_1 denotes entity for leaves
    const int entity_index = bbox[1];

    // If we have a mesh, check that the candidate is really a collision
    if (mesh)
    {
      // Get cell
      mesh::MeshEntity cell(*mesh, mesh->topology().dim(), entity_index);
      if (CollisionPredicates::collides(cell, point))
        entities.push_back(entity_index);
    }

    // Otherwise, add the candidate
    else
      entities.push_back(entity_index);
  }

  // Check both children
  else
  {
    _compute_collisions_point(tree, point, bbox[0], entities, mesh);
    _compute_collisions_point(tree, point, bbox[1], entities, mesh);
  }
}
//-----------------------------------------------------------------------------
void BoundingBoxTree::_compute_collisions_tree(
    const BoundingBoxTree& A, const BoundingBoxTree& B, int node_A, int node_B,
    std::vector<int>& entities_A, std::vector<int>& entities_B,
    const mesh::Mesh* mesh_A, const mesh::Mesh* mesh_B)
{
  // Get bounding boxes for current nodes
  const BBox& bbox_A = A._bboxes[node_A];
  const BBox& bbox_B = B._bboxes[node_B];

  // If bounding boxes don't collide, then don't search further
  if (!B.bbox_in_bbox(A.get_bbox_coordinates(node_A), node_B))
    return;

  // Check whether we've reached a leaf in A or B
  const bool is_leaf_A = is_leaf(bbox_A, node_A);
  const bool is_leaf_B = is_leaf(bbox_B, node_B);

  // If both boxes are leaves (which we know collide), then add them
  if (is_leaf_A && is_leaf_B)
  {
    // child_1 denotes entity for leaves
    const int entity_index_A = bbox_A[1];
    const int entity_index_B = bbox_B[1];

    // If we have a mesh, check that the candidate is really a collision
    if (mesh_A)
    {
      assert(mesh_B);
      mesh::MeshEntity cell_A(*mesh_A, mesh_A->topology().dim(),
                              entity_index_A);
      mesh::MeshEntity cell_B(*mesh_B, mesh_B->topology().dim(),
                              entity_index_B);
      if (CollisionPredicates::collides(cell_A, cell_B))
      {
        entities_A.push_back(entity_index_A);
        entities_B.push_back(entity_index_B);
      }
    }

    // Otherwise, add the candidate
    else
    {
      entities_A.push_back(entity_index_A);
      entities_B.push_back(entity_index_B);
    }
  }

  // If we reached the leaf in A, then descend B
  else if (is_leaf_A)
  {
    _compute_collisions_tree(A, B, node_A, bbox_B[0], entities_A, entities_B,
                             mesh_A, mesh_B);
    _compute_collisions_tree(A, B, node_A, bbox_B[1], entities_A, entities_B,
                             mesh_A, mesh_B);
  }

  // If we reached the leaf in B, then descend A
  else if (is_leaf_B)
  {
    _compute_collisions_tree(A, B, bbox_A[0], node_B, entities_A, entities_B,
                             mesh_A, mesh_B);
    _compute_collisions_tree(A, B, bbox_A[1], node_B, entities_A, entities_B,
                             mesh_A, mesh_B);
  }

  // At this point, we know neither is a leaf so descend the largest
  // tree first. Note that nodes are added in reverse order with the
  // top bounding box at the end so the largest tree (the one with the
  // the most boxes left to traverse) has the largest node number.
  else if (node_A > node_B)
  {
    _compute_collisions_tree(A, B, bbox_A[0], node_B, entities_A, entities_B,
                             mesh_A, mesh_B);
    _compute_collisions_tree(A, B, bbox_A[1], node_B, entities_A, entities_B,
                             mesh_A, mesh_B);
  }
  else
  {
    _compute_collisions_tree(A, B, node_A, bbox_B[0], entities_A, entities_B,
                             mesh_A, mesh_B);
    _compute_collisions_tree(A, B, node_A, bbox_B[1], entities_A, entities_B,
                             mesh_A, mesh_B);
  }

  // Note that cases above can be collected in fewer cases but this
  // way the logic is easier to follow.
}
//-----------------------------------------------------------------------------
int BoundingBoxTree::_compute_first_collision(const BoundingBoxTree& tree,
                                              const Eigen::Vector3d& point,
                                              int node)
{
  // Get bounding box for current node
  const BBox& bbox = tree._bboxes[node];

  // If point is not in bounding box, then don't search further
  if (!tree.point_in_bbox(point.data(), node))
    return -1;

  // If box is a leaf (which we know contains the point), then return it
  else if (is_leaf(bbox, node))
    return bbox[1]; // child_1 denotes entity for leaves

  // Check both children
  else
  {
    int c0 = _compute_first_collision(tree, point, bbox[0]);
    if (c0 >= 0)
      return c0;

    // Check second child
    int c1 = _compute_first_collision(tree, point, bbox[1]);
    if (c1 >= 0)
      return c1;
  }

  // Point not found
  return -1;
}
//-----------------------------------------------------------------------------
int BoundingBoxTree::_compute_first_entity_collision(
    const BoundingBoxTree& tree, const Eigen::Vector3d& point, int node,
    const mesh::Mesh& mesh)
{
  // Get bounding box for current node
  const BBox& bbox = tree._bboxes[node];

  // If point is not in bounding box, then don't search further
  if (!tree.point_in_bbox(point.data(), node))
    return -1;

  // If box is a leaf (which we know contains the point), then check entity
  else if (is_leaf(bbox, node))
  {
    // Get entity (child_1 denotes entity index for leaves)
    assert(tree._tdim == mesh.topology().dim());
    const int entity_index = bbox[1];
    mesh::MeshEntity cell(mesh, mesh.topology().dim(), entity_index);

    // Check entity

    if (CollisionPredicates::collides(cell, point))
      return entity_index;
  }

  // Check both children
  else
  {
    const int c0 = _compute_first_entity_collision(tree, point, bbox[0], mesh);
    if (c0 >= 0)
      return c0;

    const int c1 = _compute_first_entity_collision(tree, point, bbox[1], mesh);
    if (c1 >= 0)
      return c1;
  }

  // Point not found
  return -1;
}
//-----------------------------------------------------------------------------
void BoundingBoxTree::_compute_closest_entity(const BoundingBoxTree& tree,
                                              const Eigen::Vector3d& point,
                                              int node, const mesh::Mesh& mesh,
                                              int& closest_entity, double& R2)
{
  // Get bounding box for current node
  const BBox& bbox = tree._bboxes[node];

  // If bounding box is outside radius, then don't search further
  const double r2 = tree.compute_squared_distance_bbox(point.data(), node);
  if (r2 > R2)
    return;

  // If box is leaf (which we know is inside radius), then shrink radius
  else if (is_leaf(bbox, node))
  {
    // Get entity (child_1 denotes entity index for leaves)
    assert(tree._tdim == mesh.topology().dim());
    const int entity_index = bbox[1];
    mesh::MeshEntity cell(mesh, mesh.topology().dim(), entity_index);

    // If entity is closer than best result so far, then return it
    const double r2 = squared_distance(cell, point);
    if (r2 < R2)
    {
      closest_entity = entity_index;
      R2 = r2;
    }
  }

  // Check both children
  else
  {
    _compute_closest_entity(tree, point, bbox[0], mesh, closest_entity, R2);
    _compute_closest_entity(tree, point, bbox[1], mesh, closest_entity, R2);
  }
}
//-----------------------------------------------------------------------------
void BoundingBoxTree::_compute_closest_point(const BoundingBoxTree& tree,
                                             const Eigen::Vector3d& point,
                                             int node, int& closest_point,
                                             double& R2)
{
  // Get bounding box for current node
  const BBox& bbox = tree._bboxes[node];

  // If box is leaf, then compute distance and shrink radius
  if (is_leaf(bbox, node))
  {
    const double r2 = tree.compute_squared_distance_point(point.data(), node);
    if (r2 < R2)
    {
      closest_point = bbox[1];
      R2 = r2;
    }
  }
  else
  {
    // If bounding box is outside radius, then don't search further
    const double r2 = tree.compute_squared_distance_bbox(point.data(), node);
    if (r2 > R2)
      return;

    // Check both children
    _compute_closest_point(tree, point, bbox[0], closest_point, R2);
    _compute_closest_point(tree, point, bbox[1], closest_point, R2);
  }
}
//-----------------------------------------------------------------------------
void BoundingBoxTree::build_point_search_tree(const mesh::Mesh& mesh) const
{
  // Don't build search tree if it already exists
  if (_point_search_tree)
    return;

  LOG(INFO) << "Building point search tree to accelerate distance queries.";

  // Create list of midpoints for all cells
  const int dim = mesh.topology().dim();
  Eigen::Array<int, Eigen::Dynamic, 1> entities(mesh.num_entities(dim));
  std::iota(entities.data(), entities.data() + entities.rows(), 0);
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> midpoints
      = mesh::midpoints(mesh, dim, entities);

  std::vector<Eigen::Vector3d> points(entities.rows());
  for (std::size_t i = 0; i < points.size(); ++i)
    points[i] = midpoints.row(i);

  // Build tree
  _point_search_tree
      = std::make_unique<BoundingBoxTree>(points, mesh.geometry().dim());
}
//-----------------------------------------------------------------------------
void BoundingBoxTree::compute_bbox_of_entity(double* b,
                                             const mesh::MeshEntity& entity,
                                             int gdim)
{
  // Get bounding box coordinates
  double* xmin = b;
  double* xmax = b + gdim;

  // Get mesh entity data
  const mesh::Geometry& geometry = entity.mesh().geometry();
  const mesh::CellType entity_type
      = mesh::cell_entity_type(entity.mesh().cell_type, entity.dim());
  const int num_vertices = mesh::cell_num_entities(entity_type, 0);
  const std::int32_t* vertices = entity.entities(0);
  assert(num_vertices >= 2);

  // Get coordinates for first vertex
  const Eigen::Ref<const EigenVectorXd> x = geometry.x(vertices[0]);
  for (int j = 0; j < gdim; ++j)
    xmin[j] = xmax[j] = x[j];

  // Compute min and max over remaining vertices
  for (int i = 1; i < num_vertices; ++i)
  {
    // const double* x = geometry.x(vertices[i]);
    const Eigen::Ref<const EigenVectorXd> x = geometry.x(vertices[i]);
    for (int j = 0; j < gdim; ++j)
    {
      xmin[j] = std::min(xmin[j], x[j]);
      xmax[j] = std::max(xmax[j], x[j]);
    }
  }
}
//-----------------------------------------------------------------------------
void BoundingBoxTree::sort_points(std::size_t axis,
                                  const std::vector<Eigen::Vector3d>& points,
                                  const std::vector<int>::iterator& begin,
                                  const std::vector<int>::iterator& middle,
                                  const std::vector<int>::iterator& end)
{
  // Comparison lambda function with capture
  auto cmp = [&points, &axis](int i, int j) -> bool {
    const double* pi = points[i].data();
    const double* pj = points[j].data();
    return pi[axis] < pj[axis];
  };

  std::nth_element(begin, middle, end, cmp);
}
//-----------------------------------------------------------------------------
std::string BoundingBoxTree::str(bool verbose)
{
  std::stringstream s;
  tree_print(s, _bboxes.size() - 1);
  return s.str();
}
//-----------------------------------------------------------------------------
void BoundingBoxTree::tree_print(std::stringstream& s, int i)
{
  int dim = _bbox_coordinates.size() / _bboxes.size();
  int idx = i * dim;
  s << "[";
  for (int j = idx; j < idx + dim; ++j)
    s << _bbox_coordinates[j] << " ";
  s << "]\n";

  if (_bboxes[i][0] == i)
    s << "leaf containing entity (" << _bboxes[i][1] << ")";
  else
  {
    s << "{";
    tree_print(s, _bboxes[i][0]);
    s << ", \n";
    tree_print(s, _bboxes[i][1]);
    s << "}\n";
  }
}
//-----------------------------------------------------------------------------
void BoundingBoxTree::sort_bboxes(std::size_t axis,
                                  const std::vector<double>& leaf_bboxes,
                                  const std::vector<int>::iterator& begin,
                                  const std::vector<int>::iterator& middle,
                                  const std::vector<int>::iterator& end,
                                  int gdim)
{
  // Comparison lambda function with capture
  auto cmp = [& dim = gdim, &leaf_bboxes, &axis](int i, int j) -> bool {
    const double* bi = leaf_bboxes.data() + 2 * dim * i + axis;
    const double* bj = leaf_bboxes.data() + 2 * dim * j + axis;
    return (bi[0] + bi[dim]) < (bj[0] + bj[dim]);
  };

  std::nth_element(begin, middle, end, cmp);
}
//-----------------------------------------------------------------------------
void BoundingBoxTree::compute_bbox_of_points(
    double* bbox, std::size_t& axis, const std::vector<Eigen::Vector3d>& points,
    const std::vector<int>::iterator& begin,
    const std::vector<int>::iterator& end, int gdim)
{
  // Get coordinates for first point
  auto it = begin;
  const double* p = points[*it].data();
  for (int i = 0; i < gdim; ++i)
  {
    bbox[i] = p[i];
    bbox[i + gdim] = p[i];
  }

  // Compute min and max over remaining points
  for (; it != end; ++it)
  {
    const double* p = points[*it].data();
    for (int i = 0; i < gdim; ++i)
    {
      bbox[i] = std::min(p[i], bbox[i]);
      bbox[i + gdim] = std::max(p[i], bbox[i + gdim]);
    }
  }

  // Compute longest axis
  axis = 0;
  double max_axis = bbox[gdim] - bbox[0];
  for (int i = 1; i < gdim; ++i)
  {
    if ((bbox[gdim + i] - bbox[i]) > max_axis)
    {
      max_axis = bbox[gdim + i] - bbox[i];
      axis = i;
    }
  }
}
//-----------------------------------------------------------------------------
void BoundingBoxTree::compute_bbox_of_bboxes(
    double* bbox, std::size_t& axis, const std::vector<double>& leaf_bboxes,
    const std::vector<int>::iterator& begin,
    const std::vector<int>::iterator& end, int gdim)
{
  // Get coordinates for first box
  auto it = begin;
  const double* b = leaf_bboxes.data() + 2 * gdim * (*it);
  std::copy(b, b + 2 * gdim, bbox);

  // Compute min and max over remaining boxes
  for (; it != end; ++it)
  {
    const double* b = leaf_bboxes.data() + 2 * gdim * (*it);
    for (int i = 0; i < gdim; ++i)
      bbox[i] = std::min(bbox[i], b[i]);
    for (int i = gdim; i < 2 * gdim; ++i)
      bbox[i] = std::max(bbox[i], b[i]);
  }

  // Compute longest axis
  axis = 0;
  if (gdim == 1)
    return;

  if (gdim == 2)
  {
    const double x = bbox[2] - bbox[0];
    const double y = bbox[3] - bbox[1];
    if (y > x)
      axis = 1;
  }
  else
  {
    const double x = bbox[3] - bbox[0];
    const double y = bbox[4] - bbox[1];
    const double z = bbox[5] - bbox[2];

    if (x > y && x > z)
      return;
    else if (y > z)
      axis = 1;
    else
      axis = 2;
  }
}
//-----------------------------------------------------------------------------
double BoundingBoxTree::compute_squared_distance_point(const double* x,
                                                       int node) const
{
  const double* p = _bbox_coordinates.data() + 2 * _gdim * node;
  double d = 0.0;
  for (int i = 0; i < _gdim; ++i)
    d += (x[i] - p[i]) * (x[i] - p[i]);

  return d;
}
//-----------------------------------------------------------------------------
double BoundingBoxTree::compute_squared_distance_bbox(const double* x,
                                                      int node) const
{
  // Note: Some else-if might be in order here but I assume the
  // compiler can do a better job at optimizing/parallelizing this
  // version. This is also the way the algorithm is presented in
  // Ericsson.

  const double* b = _bbox_coordinates.data() + 2 * _gdim * node;
  double r2 = 0.0;

  for (int i = 0; i < _gdim; ++i)
  {
    if (x[i] < b[i])
      r2 += (x[i] - b[i]) * (x[i] - b[i]);
  }
  for (int i = 0; i < _gdim; ++i)
  {
    if (x[i] > b[i + _gdim])
      r2 += (x[i] - b[i + _gdim]) * (x[i] - b[i + _gdim]);
  }

  return r2;
}
//-----------------------------------------------------------------------------
bool BoundingBoxTree::bbox_in_bbox(const double* a, int node, double rtol) const
{
  const double* b = _bbox_coordinates.data() + 2 * _gdim * node;
  for (int i = 0; i < _gdim; ++i)
  {
    const double eps = rtol * (b[i + _gdim] - b[i]);
    if (b[i] - eps > a[i + _gdim] or a[i] > b[i + _gdim] + eps)
      return false;
  }
  return true;
}
//-----------------------------------------------------------------------------
bool BoundingBoxTree::point_in_bbox(const double* x, const int node,
                                    double rtol) const
{
  const double* b = _bbox_coordinates.data() + 2 * _gdim * node;
  for (int i = 0; i < _gdim; ++i)
  {
    const double eps = rtol * (b[i + _gdim] - b[i]);
    if (b[i] - eps > x[i] or x[i] > b[i + _gdim] + eps)
      return false;
  }
  return true;
}
//-----------------------------------------------------------------------------
