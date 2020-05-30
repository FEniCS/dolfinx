// Copyright (C) 2013 Anders Logg
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <array>
#include <memory>
#include <vector>

namespace dolfinx
{

// Forward declarations
namespace mesh
{
class Mesh;
} // namespace mesh

namespace geometry
{

/// Axis-Aligned bounding box binary tree. It is used to find entities
/// in a collection (often a mesh::Mesh).

class BoundingBoxTree
{

public:
  /// Constructor
  /// @param[in] mesh The mesh for building the bounding box tree
  /// @param[in] tdim The topological dimension of the mesh entities to
  ///                 by the bounding box tree for
  BoundingBoxTree(const mesh::Mesh& mesh, int tdim);

  /// Constructor
  /// @param[in] points Cloud of points to build the bounding box tree
  ///                   around
  BoundingBoxTree(const std::vector<Eigen::Vector3d>& points);

  /// Move constructor
  BoundingBoxTree(BoundingBoxTree&& tree) = default;

  /// Copy constructor
  BoundingBoxTree(const BoundingBoxTree& tree) = delete;

  /// Move assignment
  BoundingBoxTree& operator=(BoundingBoxTree&& other) = default;

  /// Destructor
  ~BoundingBoxTree() = default;

  /// Return bounding box coordinates for a given node in the tree
  /// @param[in] node The bounding box node index
  /// @return The bounding box where row(0) is the lower corner and
  ///         row(1) is the upper corner
  Eigen::Array<double, 2, 3, Eigen::RowMajor> get_bbox(int node) const;

  /// Return number of bounding boxes
  int num_bboxes() const;

  /// Topological dimension of leaf entities
  int tdim() const;

  /// Print out for debugging
  std::string str() const;

  /// Get bounding box child nodes
  /// @param[in] node The bounding box node index
  /// @return The indices of the two child nodes. For leaf nodes, index
  ///         0 is equal to the node index and index 1 is equal to the
  ///         index of the entity that the leaf box bounds,   e.g. the
  ///         index of the cell that it bounds,
  std::array<int, 2> bbox(int node) const
  {
    assert(node < (int)_bboxes.rows());
    return {_bboxes(node, 0), _bboxes(node, 1)};
  }

private:
  // Constructor
  BoundingBoxTree(
      const Eigen::Array<int, Eigen::Dynamic, 2, Eigen::RowMajor>& bboxes,
      const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>&
          bbox_coords);

  // Topological dimension of leaf entities
  int _tdim;

  // Print out recursively, for debugging
  void tree_print(std::stringstream& s, int i) const;

  // List of bounding boxes (parent-child-entity relations)
  Eigen::Array<int, Eigen::Dynamic, 2, Eigen::RowMajor> _bboxes;

  // List of bounding box coordinates
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> _bbox_coordinates;

public:
  /// Global tree for mesh ownership of each process (same on all
  /// processes)
  std::unique_ptr<BoundingBoxTree> global_tree;
};
} // namespace geometry
} // namespace dolfinx
