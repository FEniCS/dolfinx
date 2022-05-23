// Copyright (C) 2013-2021 Chris N. Richardson, Anders Logg, Garth N. Wells and
// Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "BoundingBoxTree.h"
#include "utils.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/utils.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>
#include <xtensor/xview.hpp>

using namespace dolfinx;
using namespace dolfinx::geometry;

namespace
{
//-----------------------------------------------------------------------------
std::vector<std::int32_t> range(const mesh::Mesh& mesh, int tdim)
{
  // Initialize entities of given dimension if they don't exist
  mesh.topology_mutable().create_entities(tdim);

  auto map = mesh.topology().index_map(tdim);
  assert(map);
  const std::int32_t num_entities = map->size_local() + map->num_ghosts();
  std::vector<std::int32_t> r(num_entities);
  std::iota(r.begin(), r.end(), 0);
  return r;
}
//-----------------------------------------------------------------------------
// Compute bounding box of mesh entity
std::array<std::array<double, 3>, 2>
compute_bbox_of_entity(const mesh::Mesh& mesh, int dim, std::int32_t index)
{
  // Get the geometrical indices for the mesh entity
  xtl::span<const double> xg = mesh.geometry().x();

  // FIXME: return of small dynamic array is expensive
  xtl::span<const std::int32_t> entity(&index, 1);
  const std::vector<std::int32_t> vertex_indices
      = mesh::entities_to_geometry(mesh, dim, entity, false);

  std::array<std::array<double, 3>, 2> b;
  b[0] = {xg[3 * vertex_indices.front()], xg[3 * vertex_indices.front() + 1],
          xg[3 * vertex_indices.front() + 2]};
  b[1] = b[0];

  // Compute min and max over vertices
  for (const int local_vertex : vertex_indices)
  {
    for (int j = 0; j < 3; ++j)
    {
      b[0][j] = std::min(b[0][j], xg[3 * local_vertex + j]);
      b[1][j] = std::max(b[1][j], xg[3 * local_vertex + j]);
    }
  }

  return b;
}
//-----------------------------------------------------------------------------
// Compute bounding box of bounding boxes
std::array<std::array<double, 3>, 2> compute_bbox_of_bboxes(
    const xtl::span<const std::pair<std::array<std::array<double, 3>, 2>,
                                    std::int32_t>>& leaf_bboxes)
{
  // Compute min and max over remaining boxes
  std::array<std::array<double, 3>, 2> b;
  b[0] = leaf_bboxes[0].first[0];
  b[1] = leaf_bboxes[0].first[1];
  for (auto& box : leaf_bboxes)
  {
    std::transform(box.first[0].begin(), box.first[0].end(), b[0].begin(),
                   b[0].begin(),
                   [](double a, double b) { return std::min(a, b); });
    std::transform(box.first[1].begin(), box.first[1].end(), b[1].begin(),
                   b[1].begin(),
                   [](double a, double b) { return std::max(a, b); });
  }

  return b;
}
//------------------------------------------------------------------------------
int _build_from_leaf(
    xtl::span<std::pair<std::array<std::array<double, 3>, 2>, std::int32_t>>
        leaf_bboxes,
    std::vector<std::array<int, 2>>& bboxes,
    std::vector<double>& bbox_coordinates)
{
  if (leaf_bboxes.size() == 1)
  {
    // Reached leaf

    // Get bounding box coordinates for leaf
    const std::int32_t entity_index = leaf_bboxes[0].second;
    const std::array<double, 3> b0 = leaf_bboxes[0].first[0];
    const std::array<double, 3> b1 = leaf_bboxes[0].first[1];

    // Store bounding box data
    bboxes.push_back({entity_index, entity_index});
    bbox_coordinates.insert(bbox_coordinates.end(), b0.begin(), b0.end());
    bbox_coordinates.insert(bbox_coordinates.end(), b1.begin(), b1.end());
    return bboxes.size() - 1;
  }
  else
  {
    // Compute bounding box of all bounding boxes
    std::array<std::array<double, 3>, 2> b
        = compute_bbox_of_bboxes(leaf_bboxes);
    std::array<double, 3> b0 = b[0];
    std::array<double, 3> b1 = b[1];

    // Sort bounding boxes along longest axis
    std::array<double, 3> b_diff;
    std::transform(b1.begin(), b1.end(), b0.begin(), b_diff.begin(),
                   std::minus<double>());
    const std::size_t axis = std::distance(
        b_diff.begin(), std::max_element(b_diff.begin(), b_diff.end()));

    auto middle = std::next(leaf_bboxes.begin(), leaf_bboxes.size() / 2);

    std::nth_element(leaf_bboxes.begin(), middle, leaf_bboxes.end(),
                     [axis](const auto& p0, const auto& p1) -> bool
                     {
                       const double x0 = p0.first[0][axis] + p0.first[1][axis];
                       const double x1 = p1.first[0][axis] + p1.first[1][axis];
                       return x0 < x1;
                     });

    // Split bounding boxes into two groups and call recursively
    std::array bbox{_build_from_leaf(xtl::span(leaf_bboxes.begin(), middle),
                                     bboxes, bbox_coordinates),
                    _build_from_leaf(xtl::span(middle, leaf_bboxes.end()),
                                     bboxes, bbox_coordinates)};

    // Store bounding box data. Note that root box will be added last.
    bboxes.push_back(bbox);
    bbox_coordinates.insert(bbox_coordinates.end(), b0.begin(), b0.end());
    bbox_coordinates.insert(bbox_coordinates.end(), b1.begin(), b1.end());
    return bboxes.size() - 1;
  }
}
//-----------------------------------------------------------------------------
std::pair<std::vector<std::int32_t>, std::vector<double>> build_from_leaf(
    std::vector<std::pair<std::array<std::array<double, 3>, 2>, std::int32_t>>
        leaf_bboxes)
{
  std::vector<std::array<std::int32_t, 2>> bboxes;
  std::vector<double> bbox_coordinates;
  _build_from_leaf(leaf_bboxes, bboxes, bbox_coordinates);

  std::vector<std::int32_t> bbox_array(2 * bboxes.size());
  for (std::size_t i = 0; i < bboxes.size(); ++i)
  {
    bbox_array[2 * i] = bboxes[i][0];
    bbox_array[2 * i + 1] = bboxes[i][1];
  }

  return {std::move(bbox_array), std::move(bbox_coordinates)};
}
//-----------------------------------------------------------------------------
int _build_from_point(
    xtl::span<std::pair<std::array<double, 3>, std::int32_t>> points,
    std::vector<std::array<std::int32_t, 2>>& bboxes,
    std::vector<double>& bbox_coordinates)
{
  // Reached leaf
  if (points.size() == 1)
  {
    // Store bounding box data

    // Index of entity contained in leaf
    const std::int32_t c1 = points[0].second;
    bboxes.push_back({c1, c1});
    bbox_coordinates.insert(bbox_coordinates.end(), points[0].first.begin(),
                            points[0].first.end());
    bbox_coordinates.insert(bbox_coordinates.end(), points[0].first.begin(),
                            points[0].first.end());
    return bboxes.size() - 1;
  }

  // Compute bounding box of all points
  auto minmax = std::minmax_element(points.begin(), points.end());
  std::array<double, 3> b0 = minmax.first->first;
  std::array<double, 3> b1 = minmax.second->first;

  // Sort bounding boxes along longest axis
  std::array<double, 3> b_diff;
  std::transform(b1.begin(), b1.end(), b0.begin(), b_diff.begin(),
                 std::minus<double>());
  const std::size_t axis = std::distance(
      b_diff.begin(), std::max_element(b_diff.begin(), b_diff.end()));
  auto middle = std::next(points.begin(), points.size() / 2);
  std::nth_element(
      points.begin(), middle, points.end(),
      [axis](const std::pair<std::array<double, 3>, std::int32_t>& p0,
             const std::pair<std::array<double, 3>, std::int32_t>& p1) -> bool
      { return p0.first[axis] < p1.first[axis]; });

  // Split bounding boxes into two groups and call recursively
  std::array bbox{_build_from_point(xtl::span(points.begin(), middle), bboxes,
                                    bbox_coordinates),
                  _build_from_point(xtl::span(middle, points.end()), bboxes,
                                    bbox_coordinates)};

  // Store bounding box data. Note that root box will be added last.
  bboxes.push_back(bbox);
  bbox_coordinates.insert(bbox_coordinates.end(), b0.begin(), b0.end());
  bbox_coordinates.insert(bbox_coordinates.end(), b1.begin(), b1.end());
  return bboxes.size() - 1;
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
BoundingBoxTree::BoundingBoxTree(const mesh::Mesh& mesh, int tdim,
                                 double padding)
    : BoundingBoxTree::BoundingBoxTree(mesh, tdim, range(mesh, tdim), padding)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BoundingBoxTree::BoundingBoxTree(const mesh::Mesh& mesh, int tdim,
                                 const xtl::span<const std::int32_t>& entities,
                                 double padding)
    : _tdim(tdim)
{
  if (tdim < 0 or tdim > mesh.topology().dim())
  {
    throw std::runtime_error("Dimension must be non-negative and less than or "
                             "equal to the topological dimension of the mesh");
  }

  // Initialize entities of given dimension if they don't exist
  mesh.topology_mutable().create_entities(tdim);
  mesh.topology_mutable().create_connectivity(tdim, mesh.topology().dim());

  // Create bounding boxes for all mesh entities (leaves)
  std::vector<std::pair<std::array<std::array<double, 3>, 2>, std::int32_t>>
      leaf_bboxes;
  leaf_bboxes.reserve(entities.size());
  for (std::int32_t e : entities)
  {
    std::array<std::array<double, 3>, 2> b
        = compute_bbox_of_entity(mesh, tdim, e);
    std::transform(b[0].begin(), b[0].end(), b[0].begin(),
                   [padding](double x) { return x - padding; });
    std::transform(b[1].begin(), b[1].end(), b[1].begin(),
                   [padding](double& x) { return x + padding; });
    leaf_bboxes.emplace_back(b, e);
  }

  // Recursively build the bounding box tree from the leaves
  if (!leaf_bboxes.empty())
    std::tie(_bboxes, _bbox_coordinates) = build_from_leaf(leaf_bboxes);

  LOG(INFO) << "Computed bounding box tree with " << num_bboxes()
            << " nodes for " << entities.size() << " entities.";
}
//----------------------------------------------------------------------------------
BoundingBoxTree::BoundingBoxTree(
    std::vector<std::pair<std::array<double, 3>, std::int32_t>> points)
    : _tdim(0)
{
  const std::int32_t num_leaves = points.size();

  // Recursively build the bounding box tree from the leaves
  std::vector<std::array<int, 2>> bboxes;
  if (num_leaves > 0)
  {
    _build_from_point(tcb::make_span(points), bboxes, _bbox_coordinates);
    _bboxes.resize(2 * bboxes.size());
    for (std::size_t i = 0; i < bboxes.size(); ++i)
    {
      _bboxes[2 * i] = bboxes[i][0];
      _bboxes[2 * i + 1] = bboxes[i][1];
    }
  }

  LOG(INFO) << "Computed bounding box tree with " << num_bboxes()
            << " nodes for " << num_leaves << " points.";
}
//-----------------------------------------------------------------------------
BoundingBoxTree::BoundingBoxTree(std::vector<std::int32_t>&& bboxes,
                                 std::vector<double>&& bbox_coords)
    : _tdim(0), _bboxes(bboxes), _bbox_coordinates(bbox_coords)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BoundingBoxTree BoundingBoxTree::create_global_tree(MPI_Comm comm) const
{
  // Build tree for each rank
  const int mpi_size = dolfinx::MPI::size(comm);

  // Send root node coordinates to all processes
  std::vector<double> send_bbox(6, 0.0);
  if (num_bboxes() > 0)
    std::copy_n(std::prev(_bbox_coordinates.end(), 6), 6, send_bbox.begin());
  std::vector<double> recv_bbox(mpi_size * 6);
  MPI_Allgather(send_bbox.data(), 6, MPI_DOUBLE, recv_bbox.data(), 6,
                MPI_DOUBLE, comm);

  std::vector<std::pair<std::array<std::array<double, 3>, 2>, std::int32_t>>
      _recv_bbox(mpi_size);
  for (std::size_t i = 0; i < _recv_bbox.size(); ++i)
  {
    common::impl::copy_N<3>(std::next(recv_bbox.begin(), 6 * i),
                            _recv_bbox[i].first[0].begin());
    common::impl::copy_N<3>(std::next(recv_bbox.begin(), 6 * i + 3),
                            _recv_bbox[i].first[1].begin());
    _recv_bbox[i].second = i;
  }

  auto [global_bboxes, global_coords] = build_from_leaf(_recv_bbox);
  BoundingBoxTree global_tree(std::move(global_bboxes),
                              std::move(global_coords));

  LOG(INFO) << "Computed global bounding box tree with "
            << global_tree.num_bboxes() << " boxes.";

  return global_tree;
}
//-----------------------------------------------------------------------------
std::int32_t BoundingBoxTree::num_bboxes() const { return _bboxes.size() / 2; }
//-----------------------------------------------------------------------------
std::string BoundingBoxTree::str() const
{
  std::stringstream s;
  tree_print(s, _bboxes.size() / 2 - 1);
  return s.str();
}
//-----------------------------------------------------------------------------
int BoundingBoxTree::tdim() const { return _tdim; }
//-----------------------------------------------------------------------------
void BoundingBoxTree::tree_print(std::stringstream& s, int i) const
{
  s << "[";
  for (int j = 0; j < 2; ++j)
  {
    for (int k = 0; k < 3; ++k)
      s << _bbox_coordinates[6 * i + j * 3 + k] << " ";
    if (j == 0)
      s << "]->"
        << "[";
  }
  s << "]\n";

  if (_bboxes[2 * i] == _bboxes[2 * i + 1])
    s << "leaf containing entity (" << _bboxes[2 * i + 1] << ")";
  else
  {
    s << "{";
    tree_print(s, _bboxes[2 * i]);
    s << ", \n";
    tree_print(s, _bboxes[2 * i + 1]);
    s << "}\n";
  }
}
//-----------------------------------------------------------------------------
xt::xtensor_fixed<double, xt::xshape<2, 3>>
BoundingBoxTree::get_bbox(std::size_t node) const
{
  xt::xtensor_fixed<double, xt::xshape<2, 3>> x;
  common::impl::copy_N<3>(std::next(_bbox_coordinates.begin(), 6 * node),
                          x.begin());
  common::impl::copy_N<3>(std::next(_bbox_coordinates.begin(), 6 * node + 3),
                          std::next(x.begin(), 3));
  return x;
}
//-----------------------------------------------------------------------------
