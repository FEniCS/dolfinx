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
// Last changed: 2013-05-01

#ifndef __BOUNDING_BOX_TREE_3D_H
#define __BOUNDING_BOX_TREE_3D_H

#include <vector>
#include <dolfin/common/constants.h>
#include <dolfin/log/log.h>
#include "GenericBoundingBoxTree.h"

namespace dolfin
{

  // Forward declarations
  class Mesh;
  class MeshEntity;
  class Point;

  /// This class implements a (distributed) axis aligned bounding box
  /// tree (AABB tree). Bounding box trees can be created from meshes
  /// and [other data structures, to be filled in].

  class BoundingBoxTree3D : public GenericBoundingBoxTree
  {
  public:

    /// Create bounding box tree for cells of mesh
    BoundingBoxTree3D(const Mesh& mesh);

    /// Create bounding box tree for mesh entites of given dimension
    BoundingBoxTree3D(const Mesh& mesh, unsigned int dimension);

    /// Destructor
    ~BoundingBoxTree3D();

    /// Find entities intersecting the given _Point_
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

      double xmin, xmax;
      double ymin, ymax;
      double zmin, zmax;

      // Check whether coordinate is contained in box
      inline bool contains(const double* x) const
      {
        return (xmin - DOLFIN_EPS < x[0] && x[0] < xmax + DOLFIN_EPS &&
                ymin - DOLFIN_EPS < x[1] && x[1] < ymax + DOLFIN_EPS &&
                zmin - DOLFIN_EPS < x[2] && x[2] < zmax + DOLFIN_EPS);
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
                          const std::vector<unsigned int>::iterator& end);

    // Compute bounding box of mesh entity
    void compute_bbox_of_entity(double* bbox,
                                const MeshEntity& entity) const;

    // Compute bounding box of bounding boxes (3d)
    void
    compute_bbox_of_bboxes_3d(double* bbox,
                              unsigned short int& axis,
                              const std::vector<double>& leaf_bboxes,
                              const std::vector<unsigned int>::iterator& begin,
                              const std::vector<unsigned int>::iterator& end);

    /// Find entities intersecting the given coordinate (recursive)
    void find(const double* x,
              unsigned int node,
              std::vector<unsigned int>& entities) const;

    // Geometric dimension
    unsigned int _gdim;

    // List of bounding boxes
    std::vector<BBox> bboxes;

  };

}

#endif
