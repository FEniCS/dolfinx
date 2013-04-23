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
// Last changed: 2013-04-23

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

  private:

    // Bounding box data. The 'entity' field is only set for leaves
    // and is otherwise undefined. A leaf is signified by both children
    // being set to 0.
    struct BBox
    {
      // Bounding box data
      unsigned int entity;
      unsigned int child_0;
      unsigned int child_1;
      short unsigned int axis;
      double min;
      double max;

      // Check whether coordinate is contained in box
      inline bool contains(const double* x) const
      {
        return x[axis] > min - DOLFIN_EPS && x[axis] < max + DOLFIN_EPS;
      }

      // Check whether box is a leaf
      inline bool is_leaf() const
      {
        return child_0 == 0 && child_1 == 0;
      }

    };

    // Build bounding box tree of mesh
    void build(const Mesh& mesh, unsigned int dimension);

    // Build bounding box tree (recursive, 3d)
    unsigned int build_3d(const std::vector<double>& leaf_bboxes,
                          const std::vector<unsigned int>::iterator& begin,
                          const std::vector<unsigned int>::iterator& end,
                          short unsigned int parent_axis);

    // Compute bounding box of mesh entity
    void compute_bbox_of_entity(double* bbox,
                                const MeshEntity& entity) const;

    // Compute bounding box of bounding boxes (3d)
    void
    compute_bbox_of_bboxes_3d(double* bbox,
                              const std::vector<double>& leaf_bboxes,
                              const std::vector<unsigned int>::iterator& begin,
                              const std::vector<unsigned int>::iterator& end);

    // Compute longest axis of bounding box
    short unsigned int compute_longest_axis_3d(const double* bbox) const;

    /// Find entities intersecting the given coordinate (recursive)
    void find(const double* x,
              unsigned int node,
              std::vector<unsigned int>& entities) const;

    // Geometric dimension
    unsigned int _gdim;

    // List of bounding boxes
    std::vector<BBox> bboxes;

    // List of bounding box coordinates
    std::vector<double> bbox_coordinates;

  };

}

#endif
