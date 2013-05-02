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
// First added:  2013-05-02
// Last changed: 2013-05-02

#ifndef __BOUNDING_BOX_TREE_2D_H
#define __BOUNDING_BOX_TREE_2D_H

#include <algorithm>
#include <vector>
#include <dolfin/common/constants.h>
#include "GenericBoundingBoxTree.h"

namespace dolfin
{

  // Specialization of bounding box implementation to 2D

  class BoundingBoxTree2D : public GenericBoundingBoxTree
  {
  protected:

    // Comparison operators for sorting of bounding boxes. Boxes are
    // sorted by their midpoints along the longest axis.

    struct less_x
    {
      const std::vector<double>& bboxes;
      less_x(const std::vector<double>& bboxes): bboxes(bboxes) {}

      inline bool operator()(unsigned int i, unsigned int j)
      {
        const double* bi = bboxes.data() + 4*i;
        const double* bj = bboxes.data() + 4*j;
        return bi[0] + bi[2] < bj[0] + bj[2];
      }
    };

    struct less_y
    {
      const std::vector<double>& bboxes;
      less_y(const std::vector<double>& bboxes): bboxes(bboxes) {}

      inline bool operator()(unsigned int i, unsigned int j)
      {
        const double* bi = bboxes.data() + 4*i;
        const double* bj = bboxes.data() + 4*j;
        return bi[1] + bi[3] < bj[1] + bj[3];
      }
    };

    // Check whether point is in bounding box
    bool point_in_bbox(const double* x, unsigned int node) const
    {
      const double* b = bbox_coordinates.data() + 4*node;
      return (b[0] - DOLFIN_EPS < x[0] && x[0] < b[2] + DOLFIN_EPS &&
              b[1] - DOLFIN_EPS < x[1] && x[1] < b[3] + DOLFIN_EPS);
    }

    // Compute bounding box of bounding boxes
    void compute_bbox_of_bboxes(double* bbox,
                                unsigned short int& axis,
                                const std::vector<double>& leaf_bboxes,
                                const std::vector<unsigned int>::iterator& begin,
                                const std::vector<unsigned int>::iterator& end)
    {
      typedef std::vector<unsigned int>::const_iterator iterator;

      // Get coordinates for first box
      iterator it = begin;
      const double* b = leaf_bboxes.data() + 4*(*it);
      bbox[0] = b[0];
      bbox[1] = b[1];
      bbox[2] = b[2];
      bbox[3] = b[3];

      // Compute min and max over remaining boxes
      for (; it != end; ++it)
      {
        const double* b = leaf_bboxes.data() + 4*(*it);
        if (b[0] < bbox[0]) bbox[0] = b[0];
        if (b[1] < bbox[1]) bbox[1] = b[1];
        if (b[2] > bbox[2]) bbox[2] = b[2];
        if (b[3] > bbox[3]) bbox[3] = b[3];
      }

      // Compute longest axis
      const double x = b[2] - b[0];
      const double y = b[3] - b[1];

      if (x > y)
        axis = 0;
      else
        axis = 1;
    }

    // Sort leaf bounding boxes along given axis
    void sort_bboxes(unsigned short int axis,
                     const std::vector<double>& leaf_bboxes,
                     const std::vector<unsigned int>::iterator& begin,
                     const std::vector<unsigned int>::iterator& middle,
                     const std::vector<unsigned int>::iterator& end)
    {
      if (axis == 0)
        std::nth_element(begin, middle, end, less_x(leaf_bboxes));
      else
        std::nth_element(begin, middle, end, less_y(leaf_bboxes));
    }

  };

}

#endif
