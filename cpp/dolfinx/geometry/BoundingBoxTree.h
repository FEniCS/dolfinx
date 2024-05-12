// Copyright (C) 2013-2022 Chris N. Richardson, Anders Logg, Garth N. Wells,
// Jørgen S. Dokken, Sarah Roggendorf
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <dolfinx/mesh/utils.h>
#include <mpi.h>
#include <span>
#include <string>
#include <vector>

namespace dolfinx::geometry
{
namespace impl_bb
{
//-----------------------------------------------------------------------------
// Compute bounding box of mesh entity. The bounding box is defined by (lower
// left corner, top right corner). Storage flattened row-major
template <std::floating_point T>
std::array<T, 6> compute_bbox_of_entity(const mesh::Mesh<T>& mesh, int dim,
                                        std::int32_t index)
{
  // Get the geometrical indices for the mesh entity
  std::span<const T> xg = mesh.geometry().x();

  // FIXME: return of small dynamic array is expensive
  std::span<const std::int32_t> entity(&index, 1);
  mesh.topology_mutable()->create_entity_permutations();
  const std::vector<std::int32_t> vertex_indices
      = mesh::entities_to_geometry(mesh, dim, entity);

  std::array<T, 6> b;
  auto b0 = std::span(b).template subspan<0, 3>();
  auto b1 = std::span(b).template subspan<3, 3>();
  std::copy_n(std::next(xg.begin(), 3 * vertex_indices.front()), 3, b0.begin());
  std::copy_n(std::next(xg.begin(), 3 * vertex_indices.front()), 3, b1.begin());

  // Compute min and max over vertices
  for (std::int32_t local_vertex : vertex_indices)
  {
    for (std::size_t j = 0; j < 3; ++j)
    {
      b0[j] = std::min(b0[j], xg[3 * local_vertex + j]);
      b1[j] = std::max(b1[j], xg[3 * local_vertex + j]);
    }
  }

  return b;
}
//-----------------------------------------------------------------------------
// Compute bounding box of bounding boxes. Each bounding box is defined as a
// tuple (corners, entity_index). The corners of the bounding box is flattened
// row-major as (lower left corner, top right corner).
template <std::floating_point T>
std::array<T, 6> compute_bbox_of_bboxes(
    std::span<const std::pair<std::array<T, 6>, std::int32_t>> leaf_bboxes)
{
  // Compute min and max over remaining boxes
  std::array<T, 6> b = leaf_bboxes.front().first;
  for (auto [box, _] : leaf_bboxes)
  {
    std::transform(box.cbegin(), std::next(box.cbegin(), 3), b.cbegin(),
                   b.begin(), [](auto a, auto b) { return std::min(a, b); });
    std::transform(std::next(box.cbegin(), 3), box.cend(),
                   std::next(b.cbegin(), 3), std::next(b.begin(), 3),
                   [](auto a, auto b) { return std::max(a, b); });
  }

  return b;
}
//------------------------------------------------------------------------------
template <std::floating_point T>
std::int32_t _build_from_leaf(
    std::span<std::pair<std::array<T, 6>, std::int32_t>> leaf_bboxes,
    std::vector<int>& bboxes, std::vector<T>& bbox_coordinates)
{
  if (leaf_bboxes.size() == 1)
  {
    // Reached leaf

    // Get bounding box coordinates for leaf
    const auto [b, entity_index] = leaf_bboxes.front();

    // Store bounding box data
    bboxes.push_back(entity_index);
    bboxes.push_back(entity_index);
    std::copy_n(b.begin(), 6, std::back_inserter(bbox_coordinates));
    return bboxes.size() / 2 - 1;
  }
  else
  {
    // Compute bounding box of all bounding boxes
    std::array b = compute_bbox_of_bboxes<T>(leaf_bboxes);

    // Sort bounding boxes along longest axis
    std::array<T, 3> b_diff;
    std::transform(std::next(b.cbegin(), 3), b.cend(), b.cbegin(),
                   b_diff.begin(), std::minus<T>());
    const std::size_t axis = std::distance(
        b_diff.begin(), std::max_element(b_diff.begin(), b_diff.end()));

    auto middle = std::next(leaf_bboxes.begin(), leaf_bboxes.size() / 2);
    std::nth_element(leaf_bboxes.begin(), middle, leaf_bboxes.end(),
                     [axis](auto& p0, auto& p1) -> bool
                     {
                       auto x0 = p0.first[axis] + p0.first[3 + axis];
                       auto x1 = p1.first[axis] + p1.first[3 + axis];
                       return x0 < x1;
                     });

    // Split bounding boxes into two groups and call recursively
    assert(!leaf_bboxes.empty());
    std::size_t part = leaf_bboxes.size() / 2;
    std::int32_t bbox0
        = _build_from_leaf(leaf_bboxes.first(part), bboxes, bbox_coordinates);
    std::int32_t bbox1 = _build_from_leaf(
        leaf_bboxes.last(leaf_bboxes.size() - part), bboxes, bbox_coordinates);

    // Store bounding box data. Note that root box will be added last.
    bboxes.push_back(bbox0);
    bboxes.push_back(bbox1);
    std::copy_n(b.begin(), 6, std::back_inserter(bbox_coordinates));
    return bboxes.size() / 2 - 1;
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::pair<std::vector<std::int32_t>, std::vector<T>> build_from_leaf(
    std::vector<std::pair<std::array<T, 6>, std::int32_t>>& leaf_bboxes)
{
  std::vector<std::int32_t> bboxes;
  std::vector<T> bbox_coordinates;
  impl_bb::_build_from_leaf<T>(leaf_bboxes, bboxes, bbox_coordinates);
  return {std::move(bboxes), std::move(bbox_coordinates)};
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::int32_t
_build_from_point(std::span<std::pair<std::array<T, 3>, std::int32_t>> points,
                  std::vector<std::int32_t>& bboxes,
                  std::vector<T>& bbox_coordinates)
{
  // Reached leaf
  if (points.size() == 1)
  {
    // Store bounding box data

    // Index of entity contained in leaf
    const std::int32_t c1 = points[0].second;
    bboxes.push_back(c1);
    bboxes.push_back(c1);
    bbox_coordinates.insert(bbox_coordinates.end(), points[0].first.begin(),
                            points[0].first.end());
    bbox_coordinates.insert(bbox_coordinates.end(), points[0].first.begin(),
                            points[0].first.end());
    return bboxes.size() / 2 - 1;
  }

  // Compute bounding box of all points
  auto minmax = std::minmax_element(points.begin(), points.end());
  std::array<T, 3> b0 = minmax.first->first;
  std::array<T, 3> b1 = minmax.second->first;

  // Sort bounding boxes along longest axis
  std::array<T, 3> b_diff;
  std::transform(b1.begin(), b1.end(), b0.begin(), b_diff.begin(),
                 std::minus<T>());
  const std::size_t axis = std::distance(
      b_diff.begin(), std::max_element(b_diff.begin(), b_diff.end()));

  auto middle = std::next(points.begin(), points.size() / 2);
  std::nth_element(points.begin(), middle, points.end(),
                   [axis](auto& p0, auto&& p1) -> bool
                   { return p0.first[axis] < p1.first[axis]; });

  // Split bounding boxes into two groups and call recursively
  assert(!points.empty());
  std::size_t part = points.size() / 2;
  std::int32_t bbox0
      = _build_from_point(points.first(part), bboxes, bbox_coordinates);
  std::int32_t bbox1 = _build_from_point(points.last(points.size() - part),
                                         bboxes, bbox_coordinates);

  // Store bounding box data. Note that root box will be added last.
  bboxes.push_back(bbox0);
  bboxes.push_back(bbox1);
  bbox_coordinates.insert(bbox_coordinates.end(), b0.begin(), b0.end());
  bbox_coordinates.insert(bbox_coordinates.end(), b1.begin(), b1.end());
  return bboxes.size() / 2 - 1;
}
//-----------------------------------------------------------------------------
} // namespace impl_bb

/// Axis-Aligned bounding box binary tree. It is used to find entities
/// in a collection (often a mesh::Mesh).
template <std::floating_point T>
class BoundingBoxTree
{
private:
  static std::vector<std::int32_t> range(mesh::Topology& topology, int tdim)
  {
    topology.create_entities(tdim);
    auto map = topology.index_map(tdim);
    assert(map);
    const std::int32_t num_entities = map->size_local() + map->num_ghosts();
    std::vector<std::int32_t> r(num_entities);
    std::iota(r.begin(), r.end(), 0);
    return r;
  }

public:
  /// Constructor
  /// @param[in] mesh Mesh for building the bounding box tree.
  /// @param[in] tdim Topological dimension of the mesh entities to
  /// build the bounding box tree for.
  /// @param[in] entities List of entity indices (local to process) to
  /// compute the bounding box for (may be empty, if none).
  /// @param[in] padding Value to pad (extend) the the bounding box of
  /// each entity by.
  BoundingBoxTree(const mesh::Mesh<T>& mesh, int tdim,
                  std::span<const std::int32_t> entities, double padding = 0)
      : _tdim(tdim)
  {
    if (tdim < 0 or tdim > mesh.topology()->dim())
    {
      throw std::runtime_error(
          "Dimension must be non-negative and less than or "
          "equal to the topological dimension of the mesh");
    }

    // Initialize entities of given dimension if they don't exist
    mesh.topology_mutable()->create_entities(tdim);
    mesh.topology_mutable()->create_connectivity(tdim, mesh.topology()->dim());

    // Create bounding boxes for all mesh entities (leaves)
    std::vector<std::pair<std::array<T, 6>, std::int32_t>> leaf_bboxes;
    leaf_bboxes.reserve(entities.size());
    for (std::int32_t e : entities)
    {
      std::array<T, 6> b = impl_bb::compute_bbox_of_entity(mesh, tdim, e);
      std::transform(b.cbegin(), std::next(b.cbegin(), 3), b.begin(),
                     [padding](auto x) { return x - padding; });
      std::transform(std::next(b.begin(), 3), b.end(), std::next(b.begin(), 3),
                     [padding](auto x) { return x + padding; });
      leaf_bboxes.emplace_back(b, e);
    }

    // Recursively build the bounding box tree from the leaves
    if (!leaf_bboxes.empty())
      std::tie(_bboxes, _bbox_coordinates)
          = impl_bb::build_from_leaf(leaf_bboxes);

    LOG(INFO) << "Computed bounding box tree with " << num_bboxes()
              << " nodes for " << entities.size() << " entities.";
  }

  /// Constructor
  /// @param[in] mesh The mesh for building the bounding box tree
  /// @param[in] tdim The topological dimension of the mesh entities to
  /// build the bounding box tree for
  /// @param[in] padding Value to pad (extend) the the bounding box of
  /// each entity by.
  BoundingBoxTree(const mesh::Mesh<T>& mesh, int tdim, T padding = 0)
      : BoundingBoxTree::BoundingBoxTree(
            mesh, tdim, range(mesh.topology_mutable(), tdim), padding)
  {
    // Do nothing
  }

  /// Constructor @param[in] points Cloud of points, with associated
  /// point identifier index, to build the bounding box tree around
  BoundingBoxTree(std::vector<std::pair<std::array<T, 3>, std::int32_t>> points)
      : _tdim(0)
  {
    // Recursively build the bounding box tree from the leaves
    if (!points.empty())
    {
      _bboxes.clear();
      impl_bb::_build_from_point(std::span(points), _bboxes, _bbox_coordinates);
    }

    LOG(INFO) << "Computed bounding box tree with " << num_bboxes()
              << " nodes for " << points.size() << " points.";
  }

  /// Move constructor
  BoundingBoxTree(BoundingBoxTree&& tree) = default;

  /// Copy constructor
  BoundingBoxTree(const BoundingBoxTree& tree) = delete;

  /// Move assignment
  BoundingBoxTree& operator=(BoundingBoxTree&& other) = default;

  /// Copy assignment
  BoundingBoxTree& operator=(const BoundingBoxTree& other) = default;

  /// Destructor
  ~BoundingBoxTree() = default;

  /// @brief Return bounding box coordinates for a given node in the
  /// tree,
  /// @param[in] node The bounding box node index.
  /// @return Bounding box coordinates (lower_corner, upper_corner).
  /// Shape is (2, 3), row-major storage.
  std::array<T, 6> get_bbox(std::size_t node) const
  {
    std::array<T, 6> x;
    std::copy_n(_bbox_coordinates.data() + 6 * node, 6, x.begin());
    return x;
  }

  /// Compute a global bounding tree (collective on comm)
  /// This can be used to find which process a point might have a
  /// collision with.
  /// @param[in] comm MPI Communicator for collective communication
  /// @return BoundingBoxTree where each node represents a process
  BoundingBoxTree create_global_tree(MPI_Comm comm) const
  {
    // Build tree for each rank
    const int mpi_size = dolfinx::MPI::size(comm);

    // Send root node coordinates to all processes
    // This is to counteract the fact that a process might have 0 bounding box
    // causing false positives on process collisions around (0,0,0)
    constexpr T max_val = std::numeric_limits<T>::max();
    std::array<T, 6> send_bbox
        = {max_val, max_val, max_val, max_val, max_val, max_val};
    if (num_bboxes() > 0)
      std::copy_n(std::prev(_bbox_coordinates.end(), 6), 6, send_bbox.begin());
    std::vector<T> recv_bbox(mpi_size * 6);
    MPI_Allgather(send_bbox.data(), 6, dolfinx::MPI::mpi_type<T>(),
                  recv_bbox.data(), 6, dolfinx::MPI::mpi_type<T>(), comm);

    std::vector<std::pair<std::array<T, 6>, std::int32_t>> _recv_bbox(mpi_size);
    for (std::size_t i = 0; i < _recv_bbox.size(); ++i)
    {
      std::copy_n(std::next(recv_bbox.begin(), 6 * i), 6,
                  _recv_bbox[i].first.begin());
      _recv_bbox[i].second = i;
    }

    auto [global_bboxes, global_coords] = impl_bb::build_from_leaf(_recv_bbox);
    BoundingBoxTree global_tree(std::move(global_bboxes),
                                std::move(global_coords));

    LOG(INFO) << "Computed global bounding box tree with "
              << global_tree.num_bboxes() << " boxes.";

    return global_tree;
  }

  /// Return number of bounding boxes
  std::int32_t num_bboxes() const { return _bboxes.size() / 2; }

  /// Topological dimension of leaf entities
  int tdim() const { return _tdim; }

  /// Print out for debugging
  std::string str() const
  {
    std::stringstream s;
    tree_print(s, _bboxes.size() / 2 - 1);
    return s.str();
  }

  /// Get bounding box child nodes.
  ///
  /// @param[in] node The bounding box node index
  /// @return The indices of the two child nodes. If @p node is a leaf
  /// nodes, then the values in the returned array are equal and
  /// correspond to the index of the entity that the leaf node bounds,
  /// e.g. the index of the cell that it bounds.
  std::array<std::int32_t, 2> bbox(std::size_t node) const
  {
    assert(2 * node + 1 < _bboxes.size());
    return {_bboxes[2 * node], _bboxes[2 * node + 1]};
  }

private:
  // Constructor
  BoundingBoxTree(std::vector<std::int32_t>&& bboxes,
                  std::vector<T>&& bbox_coords)
      : _tdim(0), _bboxes(bboxes), _bbox_coordinates(bbox_coords)
  {
    // Do nothing
  }

  // Topological dimension of leaf entities
  int _tdim;

  // Print out recursively, for debugging
  void tree_print(std::stringstream& s, std::int32_t i) const
  {
    s << "[";
    for (std::size_t j = 0; j < 2; ++j)
    {
      for (std::size_t k = 0; k < 3; ++k)
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

  // List of bounding boxes (parent-child-entity relations)
  std::vector<std::int32_t> _bboxes;

  // List of bounding box coordinates
  std::vector<T> _bbox_coordinates;
};
} // namespace dolfinx::geometry
