// Copyright (C) 2013 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

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
//-----------------------------------------------------------------------------
// Check whether bounding box is a leaf node
bool is_leaf(const BoundingBoxTree::BBox& bbox, int node)
{
  // Leaf nodes are marked by setting child_0 equal to the node itself
  return bbox[0] == node;
}
//-----------------------------------------------------------------------------
// Compute bounding box of mesh entity
Eigen::Array<double, 2, 3, Eigen::RowMajor>
compute_bbox_of_entity(const mesh::MeshEntity& entity)
{
  // Get mesh entity data
  const mesh::Geometry& geometry = entity.mesh().geometry();
  const mesh::CellType entity_type
      = mesh::cell_entity_type(entity.mesh().cell_type, entity.dim());
  const int num_vertices = mesh::cell_num_entities(entity_type, 0);
  const std::int32_t* vertices = entity.entities(0);
  assert(num_vertices >= 2);

  const Eigen::Vector3d x0 = geometry.x(vertices[0]);
  Eigen::Array<double, 2, 3, Eigen::RowMajor> b;
  b.row(0) = x0;
  b.row(1) = x0;

  // Compute min and max over remaining vertices
  for (int i = 1; i < num_vertices; ++i)
  {
    const Eigen::Vector3d x = geometry.x(vertices[i]);
    b.row(0) = b.row(0).min(x.transpose().array());
    b.row(1) = b.row(1).max(x.transpose().array());
  }

  return b;
}
//-----------------------------------------------------------------------------
// Compute closest entity {closest_entity, R2} (recursive)
std::pair<int, double> _compute_closest_entity(const BoundingBoxTree& tree,
                                               const Eigen::Vector3d& point,
                                               int node, const mesh::Mesh& mesh,
                                               int closest_entity, double R2)
{
  // Get bounding box for current node
  const BoundingBoxTree::BBox bbox = tree.bbox(node);

  // If bounding box is outside radius, then don't search further
  const double r2 = tree.compute_squared_distance_bbox(point, node);
  if (r2 > R2)
  {
    // If bounding box is outside radius, then don't search further
    return {closest_entity, R2};
  }
  else if (is_leaf(bbox, node))
  {
    // If box is leaf (which we know is inside radius), then shrink radius

    // Get entity (child_1 denotes entity index for leaves)
    assert(tree.tdim == mesh.topology().dim());
    const int entity_index = bbox[1];
    mesh::MeshEntity cell(mesh, mesh.topology().dim(), entity_index);

    // If entity is closer than best result so far, then return it
    const double r2 = squared_distance(cell, point);
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
std::pair<int, double> _compute_closest_point(const BoundingBoxTree& tree,
                                              const Eigen::Vector3d& point,
                                              int node, int closest_point,
                                              double R2)
{
  // Get bounding box for current node
  const BoundingBoxTree::BBox bbox = tree.bbox(node);

  // If box is leaf, then compute distance and shrink radius
  if (is_leaf(bbox, node))
  {
    const double r2 = tree.compute_squared_distance_point(point, node);
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
    const double r2 = tree.compute_squared_distance_bbox(point, node);
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

//-----------------------------------------------------------------------------
// Sort points along given axis
void sort_points(int axis, const std::vector<Eigen::Vector3d>& points,
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
// Sort leaf bounding boxes along given axis
void sort_bboxes(int axis, const std::vector<double>& leaf_bboxes,
                 const std::vector<int>::iterator& begin,
                 const std::vector<int>::iterator& middle,
                 const std::vector<int>::iterator& end)
{
  // Comparison lambda function with capture
  auto cmp = [&leaf_bboxes, &axis](int i, int j) -> bool {
    const double* bi = leaf_bboxes.data() + 6 * i + axis;
    const double* bj = leaf_bboxes.data() + 6 * j + axis;
    return (bi[0] + bi[3]) < (bj[0] + bj[3]);
  };

  std::nth_element(begin, middle, end, cmp);
}
//-----------------------------------------------------------------------------
// Compute bounding box of points
Eigen::Array<double, 2, 3, Eigen::RowMajor>
compute_bbox_of_points(const std::vector<Eigen::Vector3d>& points,
                       const std::vector<int>::iterator& begin,
                       const std::vector<int>::iterator& end)
{
  Eigen::Array<double, 2, 3, Eigen::RowMajor> b;
  b.row(0) = points[*begin];
  b.row(1) = points[*begin];
  for (auto it = begin; it != end; ++it)
  {
    const Eigen::Vector3d& p = points[*it];
    b.row(0) = b.row(0).min(p.transpose().array());
    b.row(1) = b.row(1).max(p.transpose().array());
  }

  return b;
}
//-----------------------------------------------------------------------------
// Compute bounding box of bounding boxes
Eigen::Array<double, 2, 3, Eigen::RowMajor>
compute_bbox_of_bboxes(const std::vector<double>& leaf_bboxes,
                       const std::vector<int>::iterator& begin,
                       const std::vector<int>::iterator& end)
{
  Eigen::Array<double, 2, 3, Eigen::RowMajor> b
      = Eigen::Array<double, 2, 3, Eigen::RowMajor>::Zero();

  for (int i = 0; i < 3; ++i)
  {
    b(0, i) = leaf_bboxes[6 * (*begin) + i];
    b(1, i) = leaf_bboxes[6 * (*begin) + 3 + i];
  }

  // Compute min and max over remaining boxes
  for (auto it = begin; it != end; ++it)
  {
    Eigen::Vector3d p0 = Eigen::Vector3d::Zero();
    Eigen::Vector3d p1 = Eigen::Vector3d::Zero();
    for (int i = 0; i < 3; ++i)
    {
      p0(i) = leaf_bboxes[6 * (*it) + i];
      p1(i) = leaf_bboxes[6 * (*it) + 3 + i];
    }

    b.row(0) = b.row(0).min(p0.transpose().array());
    b.row(1) = b.row(1).max(p1.transpose().array());
  }

  return b;
}
//-----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
BoundingBoxTree::BoundingBoxTree(const std::vector<double>& leaf_bboxes,
                                 const std::vector<int>::iterator& begin,
                                 const std::vector<int>::iterator& end)
    : tdim(0)
{
  _build_from_leaf(leaf_bboxes, begin, end);
}
//-----------------------------------------------------------------------------
BoundingBoxTree::BoundingBoxTree(const mesh::Mesh& mesh, int tdim) : tdim(tdim)
{
  // Check dimension
  if (tdim < 1 or tdim > mesh.topology().dim())
  {
    throw std::runtime_error("Dimension must be a number between 1 and "
                             + std::to_string(mesh.topology().dim()));
  }

  // Initialize entities of given dimension if they don't exist
  mesh.create_entities(tdim);

  // Create bounding boxes for all entities (leaves)
  const int num_leaves = mesh.num_entities(tdim);
  std::vector<double> leaf_bboxes(6 * num_leaves);
  Eigen::Map<Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>>
      _leaf_bboxes(leaf_bboxes.data(), 2 * num_leaves, 3);
  for (auto& e : mesh::MeshRange(mesh, tdim))
    _leaf_bboxes.block<2, 3>(2 * e.index(), 0) = compute_bbox_of_entity(e);

  // Create leaf partition (to be sorted)
  std::vector<int> leaf_partition(num_leaves);
  std::iota(leaf_partition.begin(), leaf_partition.end(), 0);

  // Recursively build the bounding box tree from the leaves
  _build_from_leaf(leaf_bboxes, leaf_partition.begin(), leaf_partition.end());

  LOG(INFO) << "Computed bounding box tree with " << num_bboxes()
            << " nodes for " << num_leaves << " entities.";

  // Build tree for each process
  const int mpi_size = MPI::size(mesh.mpi_comm());
  if (mpi_size > 1)
  {
    // Send root node coordinates to all processes
    std::vector<double> send_bbox(_bbox_coordinates.end() - 6,
                                  _bbox_coordinates.end());
    std::vector<double> recv_bbox;
    MPI::all_gather(mesh.mpi_comm(), send_bbox, recv_bbox);
    std::vector<int> global_leaves(mpi_size);
    std::iota(global_leaves.begin(), global_leaves.end(), 0);
    global_tree.reset(new BoundingBoxTree(recv_bbox, global_leaves.begin(),
                                          global_leaves.end()));
    LOG(INFO) << "Computed global bounding box tree with "
              << global_tree->num_bboxes() << " boxes.";
  }
}
//-----------------------------------------------------------------------------
BoundingBoxTree::BoundingBoxTree(const std::vector<Eigen::Vector3d>& points)
    : tdim(0)
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
std::pair<int, double>
BoundingBoxTree::compute_closest_entity(const Eigen::Vector3d& point,
                                        const mesh::Mesh& mesh) const
{
  // Closest entity only implemented for cells. Consider extending this.
  if (this->tdim != mesh.topology().dim())
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
  if (r == 0.0)
    return guess;

  // Call recursive find function
  std::pair<int, double> p = _compute_closest_entity(
      *this, point, num_bboxes() - 1, mesh, -1, r * r);

  // Sanity check
  assert(p.first >= 0);

  p.second = sqrt(p.second);
  return p;
}
//-----------------------------------------------------------------------------
std::pair<int, double>
BoundingBoxTree::compute_closest_point(const Eigen::Vector3d& point) const
{
  // Closest point only implemented for point cloud
  if (this->tdim != 0)
  {
    throw std::runtime_error("Cannot compute closest point. "
                             "Search tree has not been built for point cloud");
  }

  // Note that we don't compute a point search tree here... That would
  // be weird.

  // Get initial guess by picking the distance to a "random" point
  int closest_point = 0;
  double R2 = compute_squared_distance_point(point, closest_point);

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
    Eigen::Array<double, 2, 3, Eigen::RowMajor> b
        = Eigen::Array<double, 2, 3, Eigen::RowMajor>::Zero();
    for (int i = 0; i < 3; ++i)
    {
      b(0, i) = leaf_bboxes[6 * entity_index + i];
      b(1, i) = leaf_bboxes[6 * entity_index + 3 + i];
    }

    // Store bounding box data
    bbox[0] = num_bboxes(); // child_0 == node denotes a leaf
    bbox[1] = entity_index; // index of entity contained in leaf
    return add_bbox(bbox, b);
  }

  // Compute bounding box of all bounding boxes
  Eigen::Array<double, 2, 3, Eigen::RowMajor> b
      = compute_bbox_of_bboxes(leaf_bboxes, begin, end);
  Eigen::Array<double, 2, 3, Eigen::RowMajor>::Index axis;
  (b.row(1) - b.row(0)).maxCoeff(&axis);

  // Sort bounding boxes along longest axis
  std::vector<int>::iterator middle = begin + (end - begin) / 2;
  sort_bboxes(axis, leaf_bboxes, begin, middle, end);

  // Split bounding boxes into two groups and call recursively
  bbox[0] = _build_from_leaf(leaf_bboxes, begin, middle);
  bbox[1] = _build_from_leaf(leaf_bboxes, middle, end);

  // Store bounding box data. Note that root box will be added last
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
  Eigen::Array<double, 2, 3, Eigen::RowMajor> b
      = compute_bbox_of_points(points, begin, end);
  Eigen::Array<double, 2, 3, Eigen::RowMajor>::Index axis;
  (b.row(1) - b.row(0)).maxCoeff(&axis);

  // Sort bounding boxes along longest axis
  std::vector<int>::iterator middle = begin + (end - begin) / 2;
  sort_points(axis, points, begin, middle, end);

  // Split bounding boxes into two groups and call recursively
  bbox[0] = _build_from_point(points, begin, middle);
  bbox[1] = _build_from_point(points, middle, end);

  // Store bounding box data. Note that root box will be added last
  return add_bbox(bbox, b);
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
  _point_search_tree = std::make_unique<BoundingBoxTree>(points);
}
//-----------------------------------------------------------------------------
int BoundingBoxTree::add_bbox(
    const BBox& bbox, const Eigen::Array<double, 2, 3, Eigen::RowMajor>& b)
{
  // Add bounding box and coordinates
  _bboxes.push_back(bbox);
  _bbox_coordinates.insert(_bbox_coordinates.end(), b.data(), b.data() + 3);
  _bbox_coordinates.insert(_bbox_coordinates.end(), b.data() + 3, b.data() + 6);
  return _bboxes.size() - 1;
}
//-----------------------------------------------------------------------------
int BoundingBoxTree::num_bboxes() const { return _bboxes.size(); }
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
  assert(_bbox_coordinates.size() / _bboxes.size() == 3);
  int idx = i * 3;
  s << "[";
  for (int j = idx; j < idx + 3; ++j)
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
double BoundingBoxTree::compute_squared_distance_point(const Eigen::Vector3d& x,
                                                       int node) const
{
  Eigen::Map<const Eigen::Vector3d> p(_bbox_coordinates.data() + 6 * node, 3);
  return (x - p).squaredNorm();
}
//-----------------------------------------------------------------------------
double BoundingBoxTree::compute_squared_distance_bbox(const Eigen::Vector3d& x,
                                                      int node) const
{
  Eigen::Map<const Eigen::Vector3d> b0(_bbox_coordinates.data() + 6 * node, 3);
  Eigen::Map<const Eigen::Vector3d> b1(_bbox_coordinates.data() + 6 * node + 3,
                                       3);
  auto d0 = (x - b0).array();
  auto d1 = (x - b1).array();
  return (d0 > 0.0).select(0, d0).matrix().squaredNorm()
         + (d1 < 0.0).select(0, d1).matrix().squaredNorm();
}
//-----------------------------------------------------------------------------
bool BoundingBoxTree::bbox_in_bbox(
    const Eigen::Array<double, 2, 3, Eigen::RowMajor>& a, int node,
    double rtol) const
{
  Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>> b(
      _bbox_coordinates.data() + 6 * node, 2, 3);
  auto eps0 = rtol * (b.row(1) - b.row(0));
  return (b.row(0) - eps0 <= a.row(1)).all()
         and (b.row(1) + eps0 >= a.row(0)).all();
}
//-----------------------------------------------------------------------------
int BoundingBoxTree::add_point(const BBox& bbox, const Eigen::Vector3d& point)
{
  // Add bounding box
  _bboxes.push_back(bbox);

  // Add point coordinates (twice)
  _bbox_coordinates.insert(_bbox_coordinates.end(), point.data(),
                           point.data() + 3);
  _bbox_coordinates.insert(_bbox_coordinates.end(), point.data(),
                           point.data() + 3);
  return _bboxes.size() - 1;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, 2, 3, Eigen::RowMajor>
BoundingBoxTree::get_bbox_coordinates(int node) const
{
  Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>> b(
      _bbox_coordinates.data() + 6 * node, 2, 3);
  return b;
}
//-----------------------------------------------------------------------------
bool BoundingBoxTree::point_in_bbox(const Eigen::Vector3d& x, const int node,
                                    double rtol) const
{
  Eigen::Map<const Eigen::Vector3d> b0(_bbox_coordinates.data() + 6 * node, 3);
  Eigen::Map<const Eigen::Vector3d> b1(_bbox_coordinates.data() + 6 * node + 3,
                                       3);
  auto eps0 = rtol * (b1 - b0);
  return (x.array() >= (b0 - eps0).array()).all()
         and (x.array() <= (b1 + eps0).array()).all();
}
//-----------------------------------------------------------------------------
