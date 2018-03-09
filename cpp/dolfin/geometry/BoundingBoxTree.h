// Copyright (C) 2013 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <memory>
#include <sstream>
#include <vector>
#include <limits>
#include <dolfin/geometry/Point.h>

namespace dolfin
{

// Forward declarations
namespace mesh
{
class Mesh;
class MeshEntity;
}

namespace geometry
{

/// Axis-Aligned Bounding Box Tree, used to find entities in a collection
/// (often a mesh::Mesh)

class BoundingBoxTree
{
public:
  /// Constructor
  BoundingBoxTree(std::size_t gdim);

  ~BoundingBoxTree() = default;

  /// Build bounding box tree for mesh entities of given dimension
  void build(const mesh::Mesh& mesh, std::size_t tdim);

  /// Build bounding box tree for point cloud
  void build(const std::vector<Point>& points);

  /// Compute all collisions between bounding boxes and _Point_
  std::vector<unsigned int> compute_collisions(const Point& point) const;

  /// Compute all collisions between bounding boxes and _BoundingBoxTree_
  std::pair<std::vector<unsigned int>, std::vector<unsigned int>>
  compute_collisions(const BoundingBoxTree& tree) const;

  /// Compute all collisions between entities and _Point_
  std::vector<unsigned int>
  compute_entity_collisions(const Point& point, const mesh::Mesh& mesh) const;

  /// Compute all collisions between processes and _Point_
  /// returning a list of process ranks
  std::vector<unsigned int>
  compute_process_collisions(const Point& point) const;

  /// Compute all collisions between entities and _BoundingBoxTree_
  std::pair<std::vector<unsigned int>, std::vector<unsigned int>>
  compute_entity_collisions(const BoundingBoxTree& tree,
                            const mesh::Mesh& mesh_A,
                            const mesh::Mesh& mesh_B) const;

  /// Compute first collision between bounding boxes and _Point_
  unsigned int compute_first_collision(const Point& point) const;

  /// Compute first collision between entities and _Point_
  unsigned int compute_first_entity_collision(const Point& point,
                                              const mesh::Mesh& mesh) const;

  /// Compute closest entity and distance to _Point_
  std::pair<unsigned int, double>
  compute_closest_entity(const Point& point, const mesh::Mesh& mesh) const;

  /// Compute closest point and distance to _Point_
  std::pair<unsigned int, double>
  compute_closest_point(const Point& point) const;

  /// Determine if a point collides with a BoundingBox of
  /// the tree
  bool collides(const Point& point) const
  {
    return compute_first_collision(point)
           != std::numeric_limits<unsigned int>::max();
  }

  /// Determine if a point collides with an entity of the mesh
  /// (usually a cell)
  bool collides_entity(const Point& point, const mesh::Mesh& mesh) const
  {
    return compute_first_entity_collision(point, mesh)
           != std::numeric_limits<unsigned int>::max();
  }

  /// Print out for debugging
  std::string str(bool verbose = false);

private:
  // Bounding box data structure. Leaf nodes are indicated by setting child_0
  // equal to the node itself. For leaf nodes, child_1 is set to the
  // index of the entity contained in the leaf bounding box.
  struct BBox
  {
    // Child 0
    unsigned int child_0;
    // Child 1
    unsigned int child_1;
  };

  // Clear existing data if any
  void clear();

  //--- Recursive build functions ---

  // Build bounding box tree for entities (recursive)
  unsigned int _build(const std::vector<double>& leaf_bboxes,
                      const std::vector<unsigned int>::iterator& begin,
                      const std::vector<unsigned int>::iterator& end);

  // Build bounding box tree for points (recursive)
  unsigned int _build(const std::vector<Point>& points,
                      const std::vector<unsigned int>::iterator& begin,
                      const std::vector<unsigned int>::iterator& end);

  //--- Recursive search functions ---

  // Note that these functions are made static for consistency as
  // some of them need to deal with more than one tree.

  // Compute collisions with point (recursive)
  static void _compute_collisions(const BoundingBoxTree& tree,
                                  const Point& point, unsigned int node,
                                  std::vector<unsigned int>& entities,
                                  const mesh::Mesh* mesh);

  // Compute collisions with tree (recursive)
  static void _compute_collisions(const BoundingBoxTree& A,
                                  const BoundingBoxTree& B, unsigned int node_A,
                                  unsigned int node_B,
                                  std::vector<unsigned int>& entities_A,
                                  std::vector<unsigned int>& entities_B,
                                  const mesh::Mesh* mesh_A,
                                  const mesh::Mesh* mesh_B);

  // Compute first collision (recursive)
  static unsigned int _compute_first_collision(const BoundingBoxTree& tree,
                                               const Point& point,
                                               unsigned int node);

  // Compute first entity collision (recursive)
  static unsigned int
  _compute_first_entity_collision(const BoundingBoxTree& tree,
                                  const Point& point, unsigned int node,
                                  const mesh::Mesh& mesh);

  // Compute closest entity (recursive)
  static void _compute_closest_entity(const BoundingBoxTree& tree,
                                      const Point& point, unsigned int node,
                                      const mesh::Mesh& mesh,
                                      unsigned int& closest_entity, double& R2);

  // Compute closest point (recursive)
  static void _compute_closest_point(const BoundingBoxTree& tree,
                                     const Point& point, unsigned int node,
                                     unsigned int& closest_point, double& R2);

  //--- Utility functions ---

  // Compute point search tree if not already done
  void build_point_search_tree(const mesh::Mesh& mesh) const;

  // Compute bounding box of mesh entity
  void compute_bbox_of_entity(double* b, const mesh::MeshEntity& entity) const;

  // Sort points along given axis
  void sort_points(std::size_t axis, const std::vector<Point>& points,
                   const std::vector<unsigned int>::iterator& begin,
                   const std::vector<unsigned int>::iterator& middle,
                   const std::vector<unsigned int>::iterator& end);

  // Add bounding box and coordinates
  inline unsigned int add_bbox(const BBox& bbox, const double* b)
  {
    // Add bounding box
    _bboxes.push_back(bbox);

    // Add bounding box coordinates
    //    for (std::size_t i = 0; i < 2 * _gdim; ++i)
    //      _bbox_coordinates.push_back(b[i]);
    _bbox_coordinates.insert(_bbox_coordinates.end(), b, b + 2 * _gdim);

    return _bboxes.size() - 1;
  }

  // Return number of bounding boxes
  inline unsigned int num_bboxes() const { return _bboxes.size(); }

  // Add bounding box and point coordinates
  inline unsigned int add_point(const BBox& bbox, const Point& point)
  {
    // Add bounding box
    _bboxes.push_back(bbox);

    // Add point coordinates (twice)
    const double* x = point.coordinates();
    for (std::size_t i = 0; i < _gdim; ++i)
      _bbox_coordinates.push_back(x[i]);
    for (std::size_t i = 0; i < _gdim; ++i)
      _bbox_coordinates.push_back(x[i]);

    return _bboxes.size() - 1;
  }

  // Check whether bounding box is a leaf node
  inline bool is_leaf(const BBox& bbox, unsigned int node) const
  {
    // Leaf nodes are marked by setting child_0 equal to the node itself
    return bbox.child_0 == node;
  }

  // Return bounding box coordinates for node
  const double* get_bbox_coordinates(unsigned int node) const
  {
    return _bbox_coordinates.data() + 2 * _gdim * node;
  }

  // Check whether point (x) is in bounding box (node)
  bool point_in_bbox(const double* x, unsigned int node) const;

  // Check whether bounding box (a) collides with bounding box (node)
  bool bbox_in_bbox(const double* a, unsigned int node) const;

  // Compute squared distance between point and bounding box
  double compute_squared_distance_bbox(const double* x,
                                       unsigned int node) const;

  // Compute squared distance between point and point
  double compute_squared_distance_point(const double* x,
                                        unsigned int node) const;

  // Compute bounding box of bounding boxes
  void compute_bbox_of_bboxes(double* bbox, std::size_t& axis,
                              const std::vector<double>& leaf_bboxes,
                              const std::vector<unsigned int>::iterator& begin,
                              const std::vector<unsigned int>::iterator& end);

  // Compute bounding box of points
  void compute_bbox_of_points(double* bbox, std::size_t& axis,
                              const std::vector<Point>& points,
                              const std::vector<unsigned int>::iterator& begin,
                              const std::vector<unsigned int>::iterator& end);

  // Sort leaf bounding boxes along given axis
  void sort_bboxes(std::size_t axis, const std::vector<double>& leaf_bboxes,
                   const std::vector<unsigned int>::iterator& begin,
                   const std::vector<unsigned int>::iterator& middle,
                   const std::vector<unsigned int>::iterator& end);

  // Print out recursively, for debugging
  void tree_print(std::stringstream& s, unsigned int i);

  //-----------------------------------------------------------------------------

  // Topological dimension of leaf entities
  std::size_t _tdim;

  // Geometric dimension of the BBT
  std::size_t _gdim;

  // List of bounding boxes (parent-child-entity relations)
  std::vector<BBox> _bboxes;

  // List of bounding box coordinates
  std::vector<double> _bbox_coordinates;

  // Point search tree used to accelerate distance queries
  mutable std::shared_ptr<BoundingBoxTree> _point_search_tree;

  // Global tree for mesh ownership of each process (same on all processes)
  std::shared_ptr<BoundingBoxTree> _global_tree;
};
}
}
