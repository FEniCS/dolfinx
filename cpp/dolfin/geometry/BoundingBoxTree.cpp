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
// Compute bounding box of mesh entity
Eigen::Array<double, 2, 3, Eigen::RowMajor>
compute_bbox_of_entity(const mesh::MeshEntity& entity)
{
  // Get mesh entity data
  const mesh::Geometry& geometry = entity.mesh().geometry();
  const mesh::CellType entity_type
      = mesh::cell_entity_type(entity.mesh().cell_type(), entity.dim());
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
int _build_from_leaf(const std::vector<double>& leaf_bboxes,
                     const std::vector<int>::iterator partition_begin,
                     const std::vector<int>::iterator partition_end,
                     std::vector<std::array<int, 2>>& bboxes,
                     std::vector<double>& bbox_coordinates)
{
  assert(partition_begin < partition_end);

  if (partition_end - partition_begin == 1)
  {
    // Reached leaf

    // Get bounding box coordinates for leaf
    const int entity_index = *partition_begin;
    Eigen::Array<double, 2, 3, Eigen::RowMajor> b
        = Eigen::Array<double, 2, 3, Eigen::RowMajor>::Zero();
    for (int i = 0; i < 3; ++i)
    {
      b(0, i) = leaf_bboxes[6 * entity_index + i];
      b(1, i) = leaf_bboxes[6 * entity_index + 3 + i];
    }

    // Store bounding box data
    // bbox[0] = num_bboxes(); // child_0 == node denotes a leaf
    // bbox[1] = entity_index; // index of entity contained in leaf
    // return add_bbox(bbox, b);
    // return add_bbox({num_bboxes(), entity_index}, b);
    bboxes.push_back({(int)bboxes.size(), entity_index});
    bbox_coordinates.insert(bbox_coordinates.end(), b.data(), b.data() + 3);
    bbox_coordinates.insert(bbox_coordinates.end(), b.data() + 3, b.data() + 6);
    return bboxes.size() - 1;
  }
  else
  {
    // Compute bounding box of all bounding boxes
    Eigen::Array<double, 2, 3, Eigen::RowMajor> b
        = compute_bbox_of_bboxes(leaf_bboxes, partition_begin, partition_end);

    // Sort bounding boxes along longest axis
    Eigen::Array<double, 2, 3, Eigen::RowMajor>::Index axis;
    (b.row(1) - b.row(0)).maxCoeff(&axis);
    std::vector<int>::iterator partition_middle
        = partition_begin + (partition_end - partition_begin) / 2;
    std::nth_element(partition_begin, partition_middle, partition_end,
                     [&leaf_bboxes, axis](int i, int j) -> bool {
                       const double* bi = leaf_bboxes.data() + 6 * i + axis;
                       const double* bj = leaf_bboxes.data() + 6 * j + axis;
                       return (bi[0] + bi[3]) < (bj[0] + bj[3]);
                     });

    // Split bounding boxes into two groups and call recursively
    std::array<int, 2> bbox;
    bbox[0] = _build_from_leaf(leaf_bboxes, partition_begin, partition_middle,
                               bboxes, bbox_coordinates);
    bbox[1] = _build_from_leaf(leaf_bboxes, partition_middle, partition_end,
                               bboxes, bbox_coordinates);

    // Store bounding box data. Note that root box will be added last.
    bboxes.push_back(bbox);
    bbox_coordinates.insert(bbox_coordinates.end(), b.data(), b.data() + 3);
    bbox_coordinates.insert(bbox_coordinates.end(), b.data() + 3, b.data() + 6);
    return bboxes.size() - 1;
  }
}
//-----------------------------------------------------------------------------
int _build_from_point(const std::vector<Eigen::Vector3d>& points,
                      const std::vector<int>::iterator begin,
                      const std::vector<int>::iterator end,
                      std::vector<std::array<int, 2>>& bboxes,
                      std::vector<double>& bbox_coordinates)
{
  assert(begin < end);

  // Reached leaf
  if (end - begin == 1)
  {
    // Store bounding box data
    const int point_index = *begin;
    const int c0 = bboxes.size(); // child_0 == node denotes a leaf
    const int c1 = point_index;   // index of entity contained in leaf
    bboxes.push_back({c0, c1});
    bbox_coordinates.insert(bbox_coordinates.end(), points[point_index].data(),
                            points[point_index].data() + 3);
    bbox_coordinates.insert(bbox_coordinates.end(), points[point_index].data(),
                            points[point_index].data() + 3);
    return bboxes.size() - 1;
  }

  // Compute bounding box of all points
  Eigen::Array<double, 2, 3, Eigen::RowMajor> b
      = compute_bbox_of_points(points, begin, end);

  // Sort bounding boxes along longest axis
  std::vector<int>::iterator middle = begin + (end - begin) / 2;
  Eigen::Array<double, 2, 3, Eigen::RowMajor>::Index axis;
  (b.row(1) - b.row(0)).maxCoeff(&axis);
  std::nth_element(begin, middle, end, [&points, &axis](int i, int j) -> bool {
    const double* pi = points[i].data();
    const double* pj = points[j].data();
    return pi[axis] < pj[axis];
  });

  // Split bounding boxes into two groups and call recursively
  std::array<int, 2> bbox;
  bbox[0] = _build_from_point(points, begin, middle, bboxes, bbox_coordinates);
  bbox[1] = _build_from_point(points, middle, end, bboxes, bbox_coordinates);

  // Store bounding box data. Note that root box will be added last
  bboxes.push_back(bbox);
  bbox_coordinates.insert(bbox_coordinates.end(), b.data(), b.data() + 3);
  bbox_coordinates.insert(bbox_coordinates.end(), b.data() + 3, b.data() + 6);
  return bboxes.size() - 1;
}
//-----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
BoundingBoxTree::BoundingBoxTree(const std::vector<double>& leaf_bboxes,
                                 const std::vector<int>::iterator begin,
                                 const std::vector<int>::iterator end)
    : _tdim(0)
{
  std::vector<std::array<int, 2>> bboxes;
  std::vector<double> bbox_coordinates;
  _build_from_leaf(leaf_bboxes, begin, end, bboxes, bbox_coordinates);

  _bboxes.resize(bboxes.size(), 2);
  for (std::size_t i = 0; i < bboxes.size(); ++i)
  {
    _bboxes(i, 0) = bboxes[i][0];
    _bboxes(i, 1) = bboxes[i][1];
  }
  _bbox_coordinates
      = Eigen::Map<Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(
          bbox_coordinates.data(), bbox_coordinates.size() / 3, 3);
}
//-----------------------------------------------------------------------------
BoundingBoxTree::BoundingBoxTree(const mesh::Mesh& mesh, int tdim) : _tdim(tdim)
{
  // Check dimension
  if (tdim < 1 or tdim > mesh.topology().dim())
  {
    throw std::runtime_error("Dimension must be a number between 1 and "
                             + std::to_string(mesh.topology().dim()));
  }

  // Initialize entities of given dimension if they don't exist
  mesh.create_entities(tdim);

  // Create bounding boxes for all mesh entities (leaves)
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
  std::vector<std::array<int, 2>> bboxes;
  std::vector<double> bbox_coordinates;
  _build_from_leaf(leaf_bboxes, leaf_partition.begin(), leaf_partition.end(),
                   bboxes, bbox_coordinates);

  LOG(INFO) << "Computed bounding box tree with " << num_bboxes()
            << " nodes for " << num_leaves << " entities.";

  // Build tree for each process
  const int mpi_size = MPI::size(mesh.mpi_comm());
  if (mpi_size > 1)
  {
    // Send root node coordinates to all processes
    std::vector<double> send_bbox(bbox_coordinates.end() - 6,
                                  bbox_coordinates.end());
    std::vector<double> recv_bbox;
    MPI::all_gather(mesh.mpi_comm(), send_bbox, recv_bbox);
    std::vector<int> global_leaves(mpi_size);
    std::iota(global_leaves.begin(), global_leaves.end(), 0);
    global_tree.reset(new BoundingBoxTree(recv_bbox, global_leaves.begin(),
                                          global_leaves.end()));
    LOG(INFO) << "Computed global bounding box tree with "
              << global_tree->num_bboxes() << " boxes.";
  }

  _bboxes.resize(bboxes.size(), 2);
  for (std::size_t i = 0; i < bboxes.size(); ++i)
  {
    _bboxes(i, 0) = bboxes[i][0];
    _bboxes(i, 1) = bboxes[i][1];
  }
  _bbox_coordinates
      = Eigen::Map<Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(
          bbox_coordinates.data(), bbox_coordinates.size() / 3, 3);
}
//-----------------------------------------------------------------------------
BoundingBoxTree::BoundingBoxTree(const std::vector<Eigen::Vector3d>& points)
    : _tdim(0)
{
  // Create leaf partition (to be sorted)
  const int num_leaves = points.size();
  std::vector<int> leaf_partition(num_leaves);
  std::iota(leaf_partition.begin(), leaf_partition.end(), 0);

  // Recursively build the bounding box tree from the leaves
  std::vector<std::array<int, 2>> bboxes;
  std::vector<double> bbox_coordinates;
  _build_from_point(points, leaf_partition.begin(), leaf_partition.end(),
                    bboxes, bbox_coordinates);

  _bboxes.resize(bboxes.size(), 2);
  for (std::size_t i = 0; i < bboxes.size(); ++i)
  {
    _bboxes(i, 0) = bboxes[i][0];
    _bboxes(i, 1) = bboxes[i][1];
  }
  _bbox_coordinates
      = Eigen::Map<Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(
          bbox_coordinates.data(), bbox_coordinates.size() / 3, 3);

  LOG(INFO) << "Computed bounding box tree with " << num_bboxes()
            << " nodes for " << num_leaves << " points.";
}
//-----------------------------------------------------------------------------
int BoundingBoxTree::num_bboxes() const { return _bboxes.rows(); }
//-----------------------------------------------------------------------------
std::string BoundingBoxTree::str(bool verbose) const
{
  std::stringstream s;
  tree_print(s, _bboxes.rows() - 1);
  return s.str();
}
//-----------------------------------------------------------------------------
int BoundingBoxTree::tdim() const { return _tdim; }
//-----------------------------------------------------------------------------
void BoundingBoxTree::tree_print(std::stringstream& s, int i) const
{
  s << "[";
  for (int j = 0; j < 3; ++j)
    s << _bbox_coordinates(i, j) << " ";
  s << "]\n";

  if (_bboxes(i, 0) == i)
    s << "leaf containing entity (" << _bboxes(i, 1) << ")";
  else
  {
    s << "{";
    tree_print(s, _bboxes(i, 0));
    s << ", \n";
    tree_print(s, _bboxes(i, 1));
    s << "}\n";
  }
}
//-----------------------------------------------------------------------------
Eigen::Array<double, 2, 3, Eigen::RowMajor>
BoundingBoxTree::get_bbox(int node) const
{
  return _bbox_coordinates.block<2, 3>(2 * node, 0);
}
//-----------------------------------------------------------------------------
