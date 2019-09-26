// Copyright (C) 2013 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <array>
#include <memory>
#include <vector>

namespace dolfin
{

// Forward declarations
namespace mesh
{
class Mesh;
class MeshEntity;
} // namespace mesh

namespace geometry
{

/// Axis-Aligned Bounding Box Tree, used to find entities in a collection
/// (often a mesh::Mesh)

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
  std::string str(bool verbose = false);

  /// Get bounding box child nodes
  /// @param[in] node The bounding box node index
  /// @return The indices of the two child nodes. For leaf nodes, index
  ///         0 is equal to the node index and index 1 is equal to the
  ///         index of the entity that the leaf box bounds,   e.g. the
  ///         index of the cell that it bounds,
  std::array<int, 2> bbox(int node) const
  {
    assert(node < (int)_bboxes.size());
    return _bboxes[node];
  }

private:
  BoundingBoxTree(const std::vector<double>& leaf_bboxes,
                  const std::vector<int>::iterator partition_begin,
                  const std::vector<int>::iterator partition_end);

  // Topological dimension of leaf entities
  int _tdim;

  // Print out recursively, for debugging
  void tree_print(std::stringstream& s, int i);

  // List of bounding boxes (parent-child-entity relations)
  std::vector<std::array<int, 2>> _bboxes;

  // List of bounding box coordinates
  std::vector<double> _bbox_coordinates;

public:
  /// Global tree for mesh ownership of each process (same on all
  /// processes)
  std::unique_ptr<BoundingBoxTree> global_tree;
};
} // namespace geometry
} // namespace dolfin
