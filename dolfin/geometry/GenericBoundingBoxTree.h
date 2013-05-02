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
// Last changed: 2013-05-02

#ifndef __GENERIC_BOUNDING_BOX_TREE_H
#define __GENERIC_BOUNDING_BOX_TREE_H

#include <vector>
#include <dolfin/common/constants.h>

namespace dolfin
{

  class Mesh;
  class MeshEntity;
  class Point;

  /// Base class for bounding box implementations

  class GenericBoundingBoxTree
  {
  public:

    /// Constructor
    GenericBoundingBoxTree();

    /// Build bounding box tree for cells of mesh
    void build(const Mesh& mesh);

    /// Build bounding box tree for mesh entites of given dimension
    void build(const Mesh& mesh, unsigned int dimension);

    /// Destructor
    virtual ~GenericBoundingBoxTree() {}

    /// Find entities intersecting the given _Point_
    std::vector<unsigned int> find(const Point& point) const;

  protected:

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

    // List of bounding boxes
    std::vector<BBox> bboxes;

    // Build bounding box tree (recursive, 3d)
    virtual unsigned int build(const std::vector<double>& leaf_bboxes,
                               const std::vector<unsigned int>::iterator& begin,
                               const std::vector<unsigned int>::iterator& end) = 0;

  private:

    /// Find entities intersecting the given coordinate (recursive)
    void find(const double* x,
              unsigned int node,
              std::vector<unsigned int>& entities) const;

    // Compute bounding box of mesh entity
    void compute_bbox_of_entity(double* b,
                                const MeshEntity& entity,
                                unsigned int gdim) const;

  };

}

#endif
