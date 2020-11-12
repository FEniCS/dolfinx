//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "BoundingBoxTree.h"
#include "utils.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/log.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>

using namespace dolfinx;
using namespace dolfinx::geometry;

namespace
{
//-----------------------------------------------------------------------------
// Compute bounding box of mesh entity
Eigen::Array<double, 2, 3, Eigen::RowMajor>
compute_bbox_of_entity(const mesh::Mesh& mesh, int dim, std::int32_t index)
{
  // Get the geometrical indices for the mesh entity
  const int tdim = mesh.topology().dim();
  const mesh::Geometry& geometry = mesh.geometry();
  mesh.topology_mutable().create_connectivity(dim, tdim);
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> entity(1);
  entity(0, 0) = index;
  Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      vertex_indices = mesh::entities_to_geometry(mesh, dim, entity, false);
  auto entity_indices = vertex_indices.row(0);

  const Eigen::Vector3d x0 = geometry.node(entity_indices[0]);
  Eigen::Array<double, 2, 3, Eigen::RowMajor> b;
  b.row(0) = x0;
  b.row(1) = x0;
  // Compute min and max over remaining vertices
  for (int i = 1; i < entity_indices.size(); ++i)
  {
    const int local_vertex = entity_indices[i];
    const Eigen::Vector3d x = geometry.node(local_vertex);
    b.row(0) = b.row(0).min(x.transpose().array());
    b.row(1) = b.row(1).max(x.transpose().array());
  }

  return b;
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
Eigen::Array<double, 2, 3, Eigen::RowMajor> compute_bbox_of_bboxes(
    const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& leaf_bboxes,
    const std::vector<int>::iterator& begin,
    const std::vector<int>::iterator& end)
{
  Eigen::Array<double, 2, 3, Eigen::RowMajor> b
      = leaf_bboxes.block<2, 3>(2 * (*begin), 0);

  // Compute min and max over remaining boxes
  for (auto it = begin; it != end; ++it)
  {
    b.row(0) = b.row(0).min(leaf_bboxes.row(2 * (*it)));
    b.row(1) = b.row(1).max(leaf_bboxes.row(2 * (*it) + 1));
  }

  return b;
}
//------------------------------------------------------------------------------
int _build_from_leaf(
    const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& leaf_bboxes,
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
        = leaf_bboxes.block<2, 3>(2 * entity_index, 0);

    // Store bounding box data
    bboxes.push_back({entity_index, entity_index});
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
    auto partition_middle
        = partition_begin + (partition_end - partition_begin) / 2;
    std::nth_element(partition_begin, partition_middle, partition_end,
                     [&leaf_bboxes, axis](int i, int j) -> bool {
                       const double bi = leaf_bboxes(i * 2, axis)
                                         + leaf_bboxes(i * 2 + 1, axis);
                       const double bj = leaf_bboxes(j * 2, axis)
                                         + leaf_bboxes(j * 2 + 1, axis);
                       return (bi < bj);
                     });

    // Split bounding boxes into two groups and call recursively
    std::array bbox{_build_from_leaf(leaf_bboxes, partition_begin,
                                     partition_middle, bboxes,
                                     bbox_coordinates),
                    _build_from_leaf(leaf_bboxes, partition_middle,
                                     partition_end, bboxes, bbox_coordinates)};

    // Store bounding box data. Note that root box will be added last.
    bboxes.push_back(bbox);
    bbox_coordinates.insert(bbox_coordinates.end(), b.data(), b.data() + 3);
    bbox_coordinates.insert(bbox_coordinates.end(), b.data() + 3, b.data() + 6);
    return bboxes.size() - 1;
  }
}
//-----------------------------------------------------------------------------
std::tuple<Eigen::Array<int, Eigen::Dynamic, 2, Eigen::RowMajor>,
           Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>>
build_from_leaf(
    const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& leaf_bboxes)
{
  assert(leaf_bboxes.size() % 2 == 0);
  std::vector<int> partition(leaf_bboxes.rows() / 2);
  std::iota(partition.begin(), partition.end(), 0);

  std::vector<std::array<int, 2>> bboxes;
  std::vector<double> bbox_coordinates;
  _build_from_leaf(leaf_bboxes, partition.begin(), partition.end(), bboxes,
                   bbox_coordinates);

  Eigen::Array<int, Eigen::Dynamic, 2, Eigen::RowMajor> bbox_array(
      bboxes.size(), 2);
  for (std::size_t i = 0; i < bboxes.size(); ++i)
  {
    bbox_array(i, 0) = bboxes[i][0];
    bbox_array(i, 1) = bboxes[i][1];
  }
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> bbox_coord_array(
      bbox_coordinates.size() / 3, 3);
  std::copy(bbox_coordinates.begin(), bbox_coordinates.end(),
            bbox_coord_array.data());

  return {bbox_array, bbox_coord_array};
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
    const int c1 = point_index; // index of entity contained in leaf
    bboxes.push_back({c1, c1});
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
  auto middle = begin + (end - begin) / 2;
  Eigen::Array<double, 2, 3, Eigen::RowMajor>::Index axis;
  (b.row(1) - b.row(0)).maxCoeff(&axis);
  std::nth_element(begin, middle, end, [&points, &axis](int i, int j) -> bool {
    const double* pi = points[i].data();
    const double* pj = points[j].data();
    return pi[axis] < pj[axis];
  });

  // Split bounding boxes into two groups and call recursively
  std::array bbox{
      _build_from_point(points, begin, middle, bboxes, bbox_coordinates),
      _build_from_point(points, middle, end, bboxes, bbox_coordinates)};

  // Store bounding box data. Note that root box will be added last
  bboxes.push_back(bbox);
  bbox_coordinates.insert(bbox_coordinates.end(), b.data(), b.data() + 3);
  bbox_coordinates.insert(bbox_coordinates.end(), b.data() + 3, b.data() + 6);
  return bboxes.size() - 1;
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
BoundingBoxTree::BoundingBoxTree(
    const Eigen::Array<int, Eigen::Dynamic, 2, Eigen::RowMajor>& bboxes,
    const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& bbox_coords)
    : _tdim(0), _bboxes(bboxes), _bbox_coordinates(bbox_coords)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BoundingBoxTree::BoundingBoxTree(const mesh::Mesh& mesh, int tdim,
                                 double padding)
    : _tdim(tdim)
{
  // Check dimension
  if (tdim < 1 or tdim > mesh.topology().dim())
  {
    throw std::runtime_error("Dimension must be a number between 1 and "
                             + std::to_string(mesh.topology().dim()));
  }

  // Initialize entities of given dimension if they don't exist
  mesh.topology_mutable().create_entities(tdim);

  // Create bounding boxes for all mesh entities (leaves)
  auto map = mesh.topology().index_map(tdim);
  assert(map);
  const std::int32_t num_leaves = map->size_local() + map->num_ghosts();
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> leaf_bboxes(
      2 * num_leaves, 3);

  for (int e = 0; e < num_leaves; ++e)
  {
    leaf_bboxes.block<2, 3>(2 * e, 0) = compute_bbox_of_entity(mesh, tdim, e);
    leaf_bboxes.block<1, 3>(2 * e, 0) -= padding;
    leaf_bboxes.block<1, 3>(2 * e + 1, 0) += padding;
  }

  // Recursively build the bounding box tree from the leaves
  std::tie(_bboxes, _bbox_coordinates) = build_from_leaf(leaf_bboxes);

  LOG(INFO) << "Computed bounding box tree with " << num_bboxes()
            << " nodes for " << num_leaves << " entities.";
}
//-----------------------------------------------------------------------------
BoundingBoxTree::BoundingBoxTree(
    const mesh::Mesh& mesh, int tdim,
    const std::vector<std::int32_t>& entity_indices, double padding)
    : _tdim(tdim), _bboxes(0, 2), _bbox_coordinates(0, 3)
{
  // Check dimension
  if (tdim < 1 or tdim > mesh.topology().dim())
  {
    throw std::runtime_error("Dimension must be a number between 1 and "
                             + std::to_string(mesh.topology().dim()));
  }

  // Initialize entities of given dimension if they don't exist
  mesh.topology_mutable().create_entities(tdim);

  // Copy and sort indices
  std::vector<std::int32_t> entity_indices_sorted(entity_indices);
  std::sort(entity_indices_sorted.begin(), entity_indices_sorted.end());

  // Create bounding boxes for all mesh entities (leaves)
  auto map = mesh.topology().index_map(tdim);
  assert(map);
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> leaf_bboxes(
      2 * entity_indices.size(), 3);
  for (std::size_t i = 0; i < entity_indices_sorted.size(); ++i)
  {
    leaf_bboxes.block<2, 3>(2 * i, 0)
        = compute_bbox_of_entity(mesh, tdim, entity_indices_sorted[i]);
    leaf_bboxes.block<1, 3>(2 * i, 0) -= padding;
    leaf_bboxes.block<1, 3>(2 * i + 1, 0) += padding;
  }
  // Recursively build the bounding box tree from the leaves
  if (leaf_bboxes.rows() > 0)
    std::tie(_bboxes, _bbox_coordinates) = build_from_leaf(leaf_bboxes);

  // Remap leaf indices
  for (int i = 0; i < _bboxes.rows(); ++i)
  {
    if (_bboxes(i, 0) == _bboxes(i, 1))
    {
      int mapped_index = entity_indices_sorted[_bboxes(i, 0)];
      _bboxes(i, 0) = mapped_index;
      _bboxes(i, 1) = mapped_index;
    }
  }

  LOG(INFO) << "Computed bounding box tree with " << num_bboxes()
            << " nodes for " << entity_indices.size() << " entities.";
}
//----------------------------------------------------------------------------------
BoundingBoxTree BoundingBoxTree::compute_global_tree(const MPI_Comm& comm) const
{
  // Build tree for each process
  const int mpi_size = MPI::size(comm);

  // Send root node coordinates to all processes
  Eigen::Array<double, 2, 3, Eigen::RowMajor> send_bbox;
  send_bbox.setZero();
  if (num_bboxes() > 0)
    send_bbox = _bbox_coordinates.bottomRows(2);
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> recv_bbox(
      mpi_size * 2, 3);
  MPI_Allgather(send_bbox.data(), 6, MPI_DOUBLE, recv_bbox.data(), 6,
                MPI_DOUBLE, comm);

  auto [global_bboxes, global_coords] = build_from_leaf(recv_bbox);
  BoundingBoxTree global_tree(global_bboxes, global_coords);

  LOG(INFO) << "Computed global bounding box tree with "
            << global_tree.num_bboxes() << " boxes.";
  return global_tree;
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
std::string BoundingBoxTree::str() const
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
