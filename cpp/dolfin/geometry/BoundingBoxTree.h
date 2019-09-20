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
                  const std::vector<int>::iterator& end, int gdim);

public:
  /// Constructor
  BoundingBoxTree(const mesh::Mesh& mesh, int tdim);

  /// Constructor
  BoundingBoxTree(const std::vector<Eigen::Vector3d>& points, int gdim);

  /// Move constructor
  BoundingBoxTree(BoundingBoxTree&& tree) = default;

  /// Copy constructor
  BoundingBoxTree(const BoundingBoxTree& tree) = default;

  /// Move assignment
  BoundingBoxTree& operator=(BoundingBoxTree&& other) = default;

  ~BoundingBoxTree() = default;

  /// Compute all collisions between bounding boxes and _Point_
  std::vector<int> compute_collisions(const Eigen::Vector3d& point) const;

  /// Compute all collisions between bounding boxes and BoundingBoxTree
  std::pair<std::vector<int>, std::vector<int>>
  compute_collisions(const BoundingBoxTree& tree) const;

  /// Compute all collisions between entities and _Point_
  std::vector<int> compute_entity_collisions(const Eigen::Vector3d& point,
                                             const mesh::Mesh& mesh) const;

  /// Compute all collisions between processes and Point returning a
  /// list of process ranks
  std::vector<int>
  compute_process_collisions(const Eigen::Vector3d& point) const;

  /// Compute all collisions between entities and BoundingBoxTree
  std::pair<std::vector<int>, std::vector<int>>
  compute_entity_collisions(const BoundingBoxTree& tree,
                            const mesh::Mesh& mesh_A,
                            const mesh::Mesh& mesh_B) const;

  /// Compute first collision between bounding boxes and Point
  int compute_first_collision(const Eigen::Vector3d& point) const;

  /// Compute first collision between entities and Point
  int compute_first_entity_collision(const Eigen::Vector3d& point,
                                     const mesh::Mesh& mesh) const;

  /// Compute closest entity and distance to Point
  std::pair<int, double> compute_closest_entity(const Eigen::Vector3d& point,
                                                const mesh::Mesh& mesh) const;

  /// Compute closest point and distance to Point
  std::pair<int, double>
  compute_closest_point(const Eigen::Vector3d& point) const;

  /// Determine if a point collides with a BoundingBox of the tree
  bool collides(const Eigen::Vector3d& point) const
  {
    return compute_first_collision(point) >= 0;
  }

  /// Determine if a point collides with an entity of the mesh (usually
  /// a cell)
  bool collides_entity(const Eigen::Vector3d& point,
                       const mesh::Mesh& mesh) const
  {
    return compute_first_entity_collision(point, mesh) >= 0;
  }

  /// Print out for debugging
  std::string str(bool verbose = false);

  /// Bounding box data structure. Leaf nodes are indicated by setting
  /// child_0 equal to the node itself. For leaf nodes, child_1 is set to
  /// the index of the entity contained in the leaf bounding box.
  using BBox = std::array<int, 2>;

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

  //--- Recursive search functions ---

  // Note that these functions are made static for consistency as some
  // of them need to deal with more than one tree.

  // Compute collisions with point (recursive)
  static void _compute_collisions_point(const BoundingBoxTree& tree,
                                        const Eigen::Vector3d& point, int node,
                                        std::vector<int>& entities,
                                        const mesh::Mesh* mesh);

  // Compute collisions with tree (recursive)
  static void _compute_collisions_tree(const BoundingBoxTree& A,
                                       const BoundingBoxTree& B, int node_A,
                                       int node_B, std::vector<int>& entities_A,
                                       std::vector<int>& entities_B,
                                       const mesh::Mesh* mesh_A,
                                       const mesh::Mesh* mesh_B);

  // Compute first collision (recursive)
  static int _compute_first_collision(const BoundingBoxTree& tree,
                                      const Eigen::Vector3d& point, int node);

  // Compute first entity collision (recursive)
  static int _compute_first_entity_collision(const BoundingBoxTree& tree,
                                             const Eigen::Vector3d& point,
                                             int node, const mesh::Mesh& mesh);

  // Compute closest entity (recursive)
  static void _compute_closest_entity(const BoundingBoxTree& tree,
                                      const Eigen::Vector3d& point, int node,
                                      const mesh::Mesh& mesh,
                                      int& closest_entity, double& R2);

  // Compute closest point (recursive)
  static void _compute_closest_point(const BoundingBoxTree& tree,
                                     const Eigen::Vector3d& point, int node,
                                     int& closest_point, double& R2);

  //--- Utility functions ---

  // Compute point search tree if not already done
  void build_point_search_tree(const mesh::Mesh& mesh) const;

  // Compute bounding box of mesh entity
  static void compute_bbox_of_entity(double* b, const mesh::MeshEntity& entity,
                                     int gdim);

  // Sort points along given axis
  static void sort_points(std::size_t axis,
                          const std::vector<Eigen::Vector3d>& points,
                          const std::vector<int>::iterator& begin,
                          const std::vector<int>::iterator& middle,
                          const std::vector<int>::iterator& end);

  // Add bounding box and coordinates
  int add_bbox(const BBox& bbox, const double* b)
  {
    // Add bounding box and coordinates
    _bboxes.push_back(bbox);
    _bbox_coordinates.insert(_bbox_coordinates.end(), b, b + 2 * _gdim);
    return _bboxes.size() - 1;
  }

  // Return number of bounding boxes
  int num_bboxes() const { return _bboxes.size(); }

  // Add bounding box and point coordinates
  int add_point(const BBox& bbox, const Eigen::Vector3d& point)
  {
    // Add bounding box
    _bboxes.push_back(bbox);

    // Add point coordinates (twice)
    for (int i = 0; i < _gdim; ++i)
      _bbox_coordinates.push_back(point[i]);
    for (int i = 0; i < _gdim; ++i)
      _bbox_coordinates.push_back(point[i]);

    return _bboxes.size() - 1;
  }

  // Return bounding box coordinates for node
  const double* get_bbox_coordinates(int node) const
  {
    return _bbox_coordinates.data() + 2 * _gdim * node;
  }

  // Check whether point (x) is in bounding box (node)
  bool point_in_bbox(const double* x, int node, double rtol = 1e-14) const;

  // Check whether bounding box (a) collides with bounding box (node)
  bool bbox_in_bbox(const double* a, int node, double rtol = 1e-14) const;

  // Compute squared distance between point and bounding box
  double compute_squared_distance_bbox(const double* x, int node) const;

  // Compute squared distance between point and point
  double compute_squared_distance_point(const double* x, int node) const;

  // Compute bounding box of bounding boxes
  static void compute_bbox_of_bboxes(double* bbox, std::size_t& axis,
                                     const std::vector<double>& leaf_bboxes,
                                     const std::vector<int>::iterator& begin,
                                     const std::vector<int>::iterator& end,
                                     int gdim);

  // Compute bounding box of points
  static void compute_bbox_of_points(double* bbox, std::size_t& axis,
                                     const std::vector<Eigen::Vector3d>& points,
                                     const std::vector<int>::iterator& begin,
                                     const std::vector<int>::iterator& end,
                                     int gdim);

  // Sort leaf bounding boxes along given axis
  static void sort_bboxes(std::size_t axis,
                          const std::vector<double>& leaf_bboxes,
                          const std::vector<int>::iterator& begin,
                          const std::vector<int>::iterator& middle,
                          const std::vector<int>::iterator& end, int gdim);

  // Print out recursively, for debugging
  void tree_print(std::stringstream& s, int i);

  //-----------------------------------------------------------------------------

  // Topological dimension of leaf entities
  int _tdim;

  // Geometric dimension of the BBT
  int _gdim;

  // List of bounding boxes (parent-child-entity relations)
  std::vector<BBox> _bboxes;

  // List of bounding box coordinates
  std::vector<double> _bbox_coordinates;

  // Point search tree used to accelerate distance queries
  mutable std::unique_ptr<BoundingBoxTree> _point_search_tree;

  // Global tree for mesh ownership of each process (same on all
  // processes)
  std::unique_ptr<BoundingBoxTree> _global_tree;
};
} // namespace geometry
} // namespace dolfin
