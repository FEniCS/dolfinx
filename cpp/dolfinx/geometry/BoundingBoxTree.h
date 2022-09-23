// Copyright (C) 2013 Anders Logg
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <mpi.h>
#include <span>
#include <string>
#include <vector>

namespace dolfinx::mesh
{
class Mesh;
}

namespace dolfinx::geometry
{

/// Axis-Aligned bounding box binary tree. It is used to find entities
/// in a collection (often a mesh::Mesh).
class BoundingBoxTree
{

public:
  /// Constructor
  /// @param[in] mesh The mesh for building the bounding box tree
  /// @param[in] tdim The topological dimension of the mesh entities to
  /// build the bounding box tree for
  /// @param[in] entities List of entity indices (local to process) to
  /// compute the bounding box for (may be empty, if none).
  /// @param[in] padding A float perscribing how much the bounding box
  /// of each entity should be padded
  BoundingBoxTree(const mesh::Mesh& mesh, int tdim,
                  std::span<const std::int32_t> entities, double padding = 0);

  /// Constructor
  /// @param[in] mesh The mesh for building the bounding box tree
  /// @param[in] tdim The topological dimension of the mesh entities to
  /// build the bounding box tree for
  /// @param[in] padding A float perscribing how much the bounding box
  /// of each entity should be padded
  BoundingBoxTree(const mesh::Mesh& mesh, int tdim, double padding = 0);

  /// Constructor @param[in] points Cloud of points, with associated
  /// point identifier index, to build the bounding box tree around
  BoundingBoxTree(
      std::vector<std::pair<std::array<double, 3>, std::int32_t>> points);

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
  std::array<double, 6> get_bbox(std::size_t node) const
  {
    std::array<double, 6> x;
    std::copy_n(_bbox_coordinates.data() + 6 * node, 6, x.begin());
    return x;
  }

  /// Compute a global bounding tree (collective on comm)
  /// This can be used to find which process a point might have a
  /// collision with.
  /// @param[in] comm MPI Communicator for collective communication
  /// @return BoundingBoxTree where each node represents a process
  BoundingBoxTree create_global_tree(MPI_Comm comm) const;

  /// Return number of bounding boxes
  std::int32_t num_bboxes() const;

  /// Topological dimension of leaf entities
  int tdim() const;

  /// Print out for debugging
  std::string str() const;

  /// Get bounding box child nodes
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
                  std::vector<double>&& bbox_coords);

  // Topological dimension of leaf entities
  int _tdim;

  // Print out recursively, for debugging
  void tree_print(std::stringstream& s, std::int32_t i) const;

  // List of bounding boxes (parent-child-entity relations)
  std::vector<std::int32_t> _bboxes;

  // List of bounding box coordinates
  std::vector<double> _bbox_coordinates;
};
} // namespace dolfinx::geometry
