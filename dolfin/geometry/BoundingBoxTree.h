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
// First added:  2013-04-09
// Last changed: 2013-04-18

#ifndef __BOUNDING_BOX_TREE_H
#define __BOUNDING_BOX_TREE_H

#include <vector>
#include <dolfin/common/constants.h>
#include <dolfin/log/log.h>

namespace dolfin
{

  // Forward declarations
  class Mesh;
  class MeshEntity;
  class Point;

  /// This class implements a (distributed) axis aligned bounding box
  /// tree (AABB tree). Bounding box trees can be created from meshes
  /// and [other data structures, to be filled in].

  class BoundingBoxTree
  {
  public:

    /// Create bounding box tree for cells of mesh.
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh for which to compute the bounding box tree.
    BoundingBoxTree(const Mesh& mesh);

    /// Create bounding box tree for mesh entites of given dimension.
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh for which to compute the bounding box tree.
    ///     dimension (unsigned int)
    ///         The entity dimension (topological dimension) for which
    ///         to compute the bounding box tree.
    BoundingBoxTree(const Mesh& mesh, unsigned int dimension);

    /// Destructor
    ~BoundingBoxTree();

    /// Find entities intersecting the given _Point_.
    ///
    /// Note that the bounding box tree only computes a list of
    /// possible candidates since the bounding box of an object may
    /// intersect even if the object itself does not.
    ///
    /// *Returns*
    ///     std::vector<unsigned int>
    ///         A list of local indices for entities that might possibly
    ///         intersect with the given object (if any).
    ///
    /// *Arguments*
    ///     point (_Point_)
    ///         The point with which to compute the intersection.
    std::vector<unsigned int> find(const Point& point) const;

    /// Get bounding box for given node in tree.
    ///
    /// *Returns*
    ///     double*
    ///         An array of length 2*gdim holding first the gdim minimum
    ///         coordinate values and then the gdim maximum values.
    ///
    /// *Arguments*
    ///     node (unsigned int)
    ///         The number of the node (breadth-first numbering, see above).
    inline double* get_bbox(unsigned int node)
    {
      dolfin_assert(node < bbox_tree.size());
      dolfin_assert(bbox_tree[node] >= 0);
      return bbox_coordinates.data() + bbox_tree[node];
    }

    /// Get bounding box for given node in tree (const version).
    ///
    /// *Returns*
    ///     double*
    ///         An array of length 2*gdim holding first the gdim minimum
    ///         coordinate values and then the gdim maximum values.
    ///
    /// *Arguments*
    ///     node (unsigned int)
    ///         The number of the node (breadth-first numbering, see above).
    inline const double* get_bbox(unsigned int node) const
    {
      dolfin_assert(node < bbox_tree.size());
      dolfin_assert(bbox_tree[node] >= 0);
      return bbox_coordinates.data() + bbox_tree[node];
    }

    /// Check whether bounding box contains point.
    inline const bool contains(const double* x, unsigned int node) const
    {
      dolfin_assert(node < bbox_tree.size());
      const double* bbox = get_bbox(node);
      for (unsigned int j = 0; j < _gdim; ++j)
        if (x[j] < bbox[j] - DOLFIN_EPS || x[j] > bbox[j + _gdim] + DOLFIN_EPS)
          return false;
      return true;
    }

    /// Check whether node is a leaf.
    inline const bool is_leaf(unsigned int node) const
    {
      dolfin_assert(node < bbox_tree.size());
      return bbox_entities[node] != -1;
    }

  private:

    // Build bounding box tree of mesh
    void build(const Mesh& mesh, unsigned int dimension);

    // Build bounding box tree of list of bounding boxes (recursive)
    void build(const std::vector<double> leaf_bboxes,
               std::vector<unsigned int> leaf_partition,
               unsigned int begin,
               unsigned int end,
               unsigned int position,
               unsigned int& pos);

    // Compute bounding box of mesh entity
    void compute_bbox(double* bbox,
                      const MeshEntity& entity) const;

    // Compute bounding box of list of bounding boxes. Only the boxes
    // indexed by the given partition list in the range [begin, end)
    // are considered.
    void compute_bbox(double* bbox,
                      const std::vector<double> bboxes,
                      const std::vector<unsigned int> partition,
                      unsigned int begin,
                      unsigned int end) const;

    // Compute longest axis of bounding boxes
    unsigned int compute_longest_axis(const double* bbox) const;

    /// Find entities intersecting the given coordinate (recursive)
    void find(const double* x,
              unsigned int node,
              std::vector<unsigned int>& entities) const;

    // Geometric dimension
    unsigned int _gdim;

    // The tree of bounding boxes, stored as a binary tree where the
    // two children of node i are numbered 2i + 1 and 2i + 2. Each
    // node is either index into the array of bounding box coordinates
    // or -1 for nonexisting nodes.
    std::vector<int> bbox_tree;

    // Mapping from bounding boxes to entity indices. The mapping is
    // only valid for leaf nodes. Other nodes are mapped to -1.
    std::vector<int> bbox_entities;

    // A list of coordinates for all bounding boxes. Each bounding box
    // is stored as 2*_gdim doubles in the list. These numbers are
    // first the _gdim minimum coordinate values and then the _gdim
    // corresponding maximum coordinate values for each axis.
    std::vector<double> bbox_coordinates;

  };

}

#endif
