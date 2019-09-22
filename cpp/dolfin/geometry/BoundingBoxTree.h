// Copyright (C) 2013 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <array>
#include <limits>
#include <memory>
#include <sstream>
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
private:
  BoundingBoxTree(const std::vector<double>& leaf_bboxes,
                  const std::vector<int>::iterator& begin,
                  const std::vector<int>::iterator& end);

public:
  /// Constructor
  BoundingBoxTree(const mesh::Mesh& mesh, int tdim);

  /// Constructor
  BoundingBoxTree(const std::vector<Eigen::Vector3d>& points);

  /// Move constructor
  BoundingBoxTree(BoundingBoxTree&& tree) = default;

  /// Copy constructor
  BoundingBoxTree(const BoundingBoxTree& tree) = default;

  /// Move assignment
  BoundingBoxTree& operator=(BoundingBoxTree&& other) = default;

  ~BoundingBoxTree() = default;

  /// Compute closest entity and distance to Point
  std::pair<int, double> compute_closest_entity(const Eigen::Vector3d& point,
                                                const mesh::Mesh& mesh) const;

  /// Compute closest point and distance to Point
  std::pair<int, double>
  compute_closest_point(const Eigen::Vector3d& point) const;

  /// Return bounding box coordinates for node
  Eigen::Array<double, 2, 3, Eigen::RowMajor>
  get_bbox_coordinates(int node) const;

  /// Check whether point (x) is in bounding box (node)
  bool point_in_bbox(const Eigen::Vector3d& x, int node,
                     double rtol = 1e-14) const;

  /// Check whether bounding box (a) collides with bounding box (node)
  bool bbox_in_bbox(const Eigen::Array<double, 2, 3, Eigen::RowMajor>& a,
                    int node, double rtol = 1e-14) const;

  /// Compute squared distance between point and bounding box
  double compute_squared_distance_bbox(const Eigen::Vector3d& x,
                                       int node) const;

  /// Compute squared distance between point and point
  double compute_squared_distance_point(const Eigen::Vector3d& x,
                                        int node) const;

  /// Print out for debugging
  std::string str(bool verbose = false);

  /// Bounding box data structure. Leaf nodes are indicated by setting
  /// child_0 equal to the node itself. For leaf nodes, child_1 is set to
  /// the index of the entity contained in the leaf bounding box.
  using BBox = std::array<int, 2>;

  /// Return bounding box
  BBox bbox(int node) const
  {
    assert(node < (int)_bboxes.size());
    return _bboxes[node];
  }

  /// Topological dimension of leaf entities
  const int tdim;

private:
  //--- Recursive build functions ---

  // Build bounding box tree for entities (recursive)
  int _build_from_leaf(const std::vector<double>& leaf_bboxes,
                       const std::vector<int>::iterator& begin,
                       const std::vector<int>::iterator& end);

  // Build bounding box tree for points (recursive)
  int _build_from_point(const std::vector<Eigen::Vector3d>& points,
                        const std::vector<int>::iterator& begin,
                        const std::vector<int>::iterator& end);

  //--- Utility functions ---

  // Compute point search tree if not already done
  void build_point_search_tree(const mesh::Mesh& mesh) const;

  // Add bounding box and coordinates
  int add_bbox(const BBox& bbox,
               const Eigen::Array<double, 2, 3, Eigen::RowMajor>& b);

public:
  /// Return number of bounding boxes
  int num_bboxes() const;

private:
  // Add bounding box and point coordinates
  int add_point(const BBox& bbox, const Eigen::Vector3d& point);

  // Print out recursively, for debugging
  void tree_print(std::stringstream& s, int i);

  // List of bounding boxes (parent-child-entity relations)
  std::vector<BBox> _bboxes;

  // List of bounding box coordinates
  std::vector<double> _bbox_coordinates;

  // Point search tree used to accelerate distance queries
  mutable std::unique_ptr<BoundingBoxTree> _point_search_tree;

public:
  /// Global tree for mesh ownership of each process (same on all
  /// processes)
  std::unique_ptr<BoundingBoxTree> global_tree;
};
} // namespace geometry
} // namespace dolfin
