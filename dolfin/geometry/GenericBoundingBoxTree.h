// Copyright (C) 2013 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2013-04-23
// Last changed: 2014-02-06

#ifndef __GENERIC_BOUNDING_BOX_TREE_H
#define __GENERIC_BOUNDING_BOX_TREE_H

#include <memory>
#include <sstream>
#include <set>
#include <vector>
#include <dolfin/geometry/Point.h>

namespace dolfin
{

  // Forward declarations
  class Mesh;
  class MeshEntity;

  /// Base class for bounding box implementations (envelope-letter
  /// design)

  class GenericBoundingBoxTree
  {
  public:

    /// Constructor
    GenericBoundingBoxTree();

    /// Destructor
    virtual ~GenericBoundingBoxTree() {}

    /// Factory function returning (empty) tree of appropriate dimension
    static std::shared_ptr<GenericBoundingBoxTree> create(unsigned int dim);

    /// Build bounding box tree for mesh entities of given dimension
    void build(const Mesh& mesh, std::size_t tdim);

    /// Build bounding box tree for point cloud
    void build(const std::vector<Point>& points);

    /// Compute all collisions between bounding boxes and _Point_
    std::vector<unsigned int>
    compute_collisions(const Point& point) const;

    /// Compute all collisions between bounding boxes and _BoundingBoxTree_
    std::pair<std::vector<unsigned int>, std::vector<unsigned int> >
    compute_collisions(const GenericBoundingBoxTree& tree) const;

    /// Compute all collisions between entities and _Point_
    std::vector<unsigned int>
    compute_entity_collisions(const Point& point,
                              const Mesh& mesh) const;

    /// Compute all collisions between processes and _Point_
    std::vector<unsigned int>
    compute_process_collisions(const Point& point) const;

    /// Compute all collisions between entities and _BoundingBoxTree_
    std::pair<std::vector<unsigned int>, std::vector<unsigned int> >
    compute_entity_collisions(const GenericBoundingBoxTree& tree,
                              const Mesh& mesh_A, const Mesh& mesh_B) const;

    /// Compute first collision between bounding boxes and _Point_
    unsigned int compute_first_collision(const Point& point) const;

    /// Compute first collision between entities and _Point_
    unsigned int compute_first_entity_collision(const Point& point,
                                              const Mesh& mesh) const;

    /// Compute closest entity and distance to _Point_
    std::pair<unsigned int, double> compute_closest_entity(const Point& point,
                                                           const Mesh& mesh) const;

    /// Compute closest point and distance to _Point_
    std::pair<unsigned int, double> compute_closest_point(const Point& point) const;

    /// Print out for debugging
    std::string str(bool verbose=false);

  protected:

    // Bounding box data. Leaf nodes are indicated by setting child_0
    // equal to the node itself. For leaf nodes, child_1 is set to the
    // index of the entity contained in the leaf bounding box.
    struct BBox
    {
      unsigned int child_0;
      unsigned int child_1;
    };

    // Topological dimension of leaf entities
    std::size_t _tdim;

    // List of bounding boxes (parent-child-entity relations)
    std::vector<BBox> _bboxes;

    // List of bounding box coordinates
    std::vector<double> _bbox_coordinates;

    // Point search tree used to accelerate distance queries
    mutable std::shared_ptr<GenericBoundingBoxTree> _point_search_tree;

    // Global tree for mesh ownership of each process (same on all processes)
    std::shared_ptr<GenericBoundingBoxTree> _global_tree;

    // Clear existing data if any
    void clear();

    //--- Recursive build functions ---

    // Build bounding box tree for entities (recursive)
    unsigned int _build(const std::vector<double>& leaf_bboxes,
                        const std::vector<unsigned int>::iterator& begin,
                        const std::vector<unsigned int>::iterator& end,
                        std::size_t gdim);

    // Build bounding box tree for points (recursive)
    unsigned int _build(const std::vector<Point>& points,
                        const std::vector<unsigned int>::iterator& begin,
                        const std::vector<unsigned int>::iterator& end,
                        std::size_t gdim);

    //--- Recursive search functions ---

    // Note that these functions are made static for consistency as
    // some of them need to deal with more than tree.

    /// Compute collisions with point (recursive)
    static void
    _compute_collisions(const GenericBoundingBoxTree& tree,
                        const Point& point,
                        unsigned int node,
                        std::vector<unsigned int>& entities,
                        const Mesh* mesh);

    /// Compute collisions with tree (recursive)
    static void
    _compute_collisions(const GenericBoundingBoxTree& A,
                        const GenericBoundingBoxTree& B,
                        unsigned int node_A,
                        unsigned int node_B,
                        std::vector<unsigned int>& entities_A,
                        std::vector<unsigned int>& entities_B,
                        const Mesh* mesh_A,
                        const Mesh* mesh_B);

    /// Compute first collision (recursive)
    static unsigned int
    _compute_first_collision(const GenericBoundingBoxTree& tree,
                             const Point& point,
                             unsigned int node);

    /// Compute first entity collision (recursive)
    static unsigned int
    _compute_first_entity_collision(const GenericBoundingBoxTree& tree,
                                    const Point& point,
                                    unsigned int node,
                                    const Mesh& mesh);

    /// Compute closest entity (recursive)
    static void _compute_closest_entity(const GenericBoundingBoxTree& tree,
                                        const Point& point,
                                        unsigned int node,
                                        const Mesh& mesh,
                                        unsigned int& closest_entity,
                                        double& R2);

    /// Compute closest point (recursive)
    static void
    _compute_closest_point(const GenericBoundingBoxTree& tree,
                           const Point& point,
                           unsigned int node,
                           unsigned int& closest_point,
                           double& R2);

    //--- Utility functions ---

    // Compute point search tree if not already done
    void build_point_search_tree(const Mesh& mesh) const;

    // Compute bounding box of mesh entity
    void compute_bbox_of_entity(double* b,
                                const MeshEntity& entity,
                                std::size_t gdim) const;

    // Sort points along given axis
    void sort_points(std::size_t axis,
                     const std::vector<Point>& points,
                     const std::vector<unsigned int>::iterator& begin,
                     const std::vector<unsigned int>::iterator& middle,
                     const std::vector<unsigned int>::iterator& end);

    // Add bounding box and coordinates
    inline unsigned int add_bbox(const BBox& bbox,
                                 const double* b,
                                 std::size_t gdim)
    {
      // Add bounding box
      _bboxes.push_back(bbox);

      // Add bounding box coordinates
      for (std::size_t i = 0; i < 2*gdim; ++i)
        _bbox_coordinates.push_back(b[i]);

      return _bboxes.size() - 1;
    }

    // Return bounding box for given node
    inline const BBox& get_bbox(unsigned int node) const
    {
      return _bboxes[node];
    }

    // Return number of bounding boxes
    inline unsigned int num_bboxes() const
    {
      return _bboxes.size();
    }

    // Add bounding box and point coordinates
    inline unsigned int add_point(const BBox& bbox,
                                  const Point& point,
                                  std::size_t gdim)
    {
      // Add bounding box
      _bboxes.push_back(bbox);

      // Add point coordinates (twice)
      const double* x = point.coordinates();
      for (std::size_t i = 0; i < gdim; ++i)
        _bbox_coordinates.push_back(x[i]);
      for (std::size_t i = 0; i < gdim; ++i)
        _bbox_coordinates.push_back(x[i]);

      return _bboxes.size() - 1;
    }

    // Check whether bounding box is a leaf node
    inline bool is_leaf(const BBox& bbox, unsigned int node) const
    {
      // Leaf nodes are marked by setting child_0 equal to the node itself
      return bbox.child_0 == node;
    }

    // Comparison operators for sorting of points. The corresponding
    // comparison operators for bounding boxes are dimension-dependent
    // and are therefore implemented in the subclasses.

    struct less_x_point
    {
      const std::vector<Point>& points;
      less_x_point(const std::vector<Point>& points): points(points) {}

      inline bool operator()(unsigned int i, unsigned int j)
      {
        const double* pi = points[i].coordinates();
        const double* pj = points[j].coordinates();
        return pi[0] < pj[0];
      }
    };

    struct less_y_point
    {
      const std::vector<Point>& points;
      less_y_point(const std::vector<Point>& points): points(points) {}

      inline bool operator()(unsigned int i, unsigned int j)
      {
        const double* pi = points[i].coordinates();
        const double* pj = points[j].coordinates();
        return pi[1] < pj[1];
      }
    };

    struct less_z_point
    {
      const std::vector<Point>& points;
      less_z_point(const std::vector<Point>& points): points(points) {}

      inline bool operator()(unsigned int i, unsigned int j)
      {
        const double* pi = points[i].coordinates();
        const double* pj = points[j].coordinates();
        return pi[2] < pj[2];
      }
    };

    //--- Dimension-dependent functions to be implemented by subclass ---

    // Return geometric dimension
    virtual std::size_t gdim() const = 0;

    // Return bounding box coordinates for node
    virtual const double* get_bbox_coordinates(unsigned int node) const = 0;

    // Check whether point (x) is in bounding box (node)
    virtual bool
    point_in_bbox(const double* x, unsigned int node) const = 0;

    // Check whether bounding box (a) collides with bounding box (node)
    virtual bool
    bbox_in_bbox(const double* a, unsigned int node) const = 0;

    // Compute squared distance between point and bounding box
    virtual double
    compute_squared_distance_bbox(const double* x, unsigned int node) const = 0;

    // Compute squared distance between point and point
    virtual double
    compute_squared_distance_point(const double* x, unsigned int node) const = 0;

    // Compute bounding box of bounding boxes
    virtual void
    compute_bbox_of_bboxes(double* bbox,
                           std::size_t& axis,
                           const std::vector<double>& leaf_bboxes,
                           const std::vector<unsigned int>::iterator& begin,
                           const std::vector<unsigned int>::iterator& end) = 0;

    // Compute bounding box of points
    virtual void
    compute_bbox_of_points(double* bbox,
                           std::size_t& axis,
                           const std::vector<Point>& points,
                           const std::vector<unsigned int>::iterator& begin,
                           const std::vector<unsigned int>::iterator& end) = 0;

    // Sort leaf bounding boxes along given axis
    virtual void
    sort_bboxes(std::size_t axis,
                const std::vector<double>& leaf_bboxes,
                const std::vector<unsigned int>::iterator& begin,
                const std::vector<unsigned int>::iterator& middle,
                const std::vector<unsigned int>::iterator& end) = 0;

    // Print out recursively, for debugging
    void tree_print(std::stringstream& s, unsigned int i);


  };

}

#endif
