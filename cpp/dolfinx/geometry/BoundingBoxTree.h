// Copyright (C) 2013 Anders Logg
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <array>
#include <dolfinx/common/MPI.h>
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
  ///                 build the bounding box tree for
  /// @param[in] entity_indices List of entity indices to compute the bounding
  /// box for (may be empty, if none).
  /// @param[in] padding A float perscribing how much the bounding box of
  /// each entity should be padded
  BoundingBoxTree(const mesh::Mesh& mesh, int tdim,
                  const std::vector<std::int32_t>& entity_indices,
                  double padding = 0);

  /// Constructor
  /// @param[in] mesh The mesh for building the bounding box tree
  /// @param[in] tdim The topological dimension of the mesh entities to
  ///                 build the bounding box tree for
  /// @param[in] padding A float perscribing how much the bounding box of
  /// each entity should be padded
  BoundingBoxTree(const mesh::Mesh& mesh, int tdim, double padding = 0);

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

  /// Compute a global bounding tree (collective on comm)
  /// This can be used to find which process a point might have a collision
  /// with.
  /// @param[in] comm MPI Communicator for collective communication
  /// @return BoundingBoxTree where each node represents a process
  BoundingBoxTree compute_global_tree(const MPI_Comm& comm) const;

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
};
} // namespace geometry
} // namespace dolfinx
