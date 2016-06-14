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
// Last changed: 2014-02-24

#ifndef __BOUNDING_BOX_TREE_1D_H
#define __BOUNDING_BOX_TREE_1D_H

#include <algorithm>
#include <vector>
#include <dolfin/common/constants.h>
#include "GenericBoundingBoxTree.h"

namespace dolfin
{

  /// Specialization of bounding box implementation to 1D

  class BoundingBoxTree1D : public GenericBoundingBoxTree
  {
  protected:

    /// Comparison operator for sorting of bounding boxes. Boxes are
    /// sorted by their midpoints along the longest axis.
    struct less_x
    {
      const std::vector<double>& bboxes;

      less_x(const std::vector<double>& bboxes): bboxes(bboxes) {}

      inline bool operator()(unsigned int i, unsigned int j)
      {
        const double* bi = bboxes.data() + 2*i;
        const double* bj = bboxes.data() + 2*j;
        return bi[0] + bi[1] < bj[0] + bj[1];
      }
    };

    /// Return geometric dimension
    std::size_t gdim() const { return 1; }

    /// Return bounding box coordinates for node
    const double* get_bbox_coordinates(unsigned int node) const
    {
      return _bbox_coordinates.data() + 2*node;
    }

    /// Check whether point (x) is in bounding box (node)
    bool point_in_bbox(const double* x, unsigned int node) const
    {
      const double* b = _bbox_coordinates.data() + 2*node;
      const double eps = DOLFIN_EPS_LARGE*(b[1] - b[0]);
      return b[0] - eps <= x[0] && x[0] <= b[1] + eps;
    }

    /// Check whether bounding box (a) collides with bounding box (node)
    bool bbox_in_bbox(const double* a, unsigned int node) const
    {
      const double* b = _bbox_coordinates.data() + 2*node;
      const double eps = DOLFIN_EPS_LARGE*(b[1] - b[0]);
      return b[0] - eps <= a[1] && a[0] <= b[1] + eps;
    }

    /// Compute squared distance between point and bounding box
    double compute_squared_distance_bbox(const double* x,
                                         unsigned int node) const
    {
      // Note: Some else-if might be in order here but I assume the
      // compiler can do a better job at optimizing/parallelizing this
      // version. This is also the way the algorithm is presented in
      // Ericsson.

      const double* b = _bbox_coordinates.data() + 2*node;
      double r2 = 0.0;

      if (x[0] < b[0]) r2 += (x[0] - b[0])*(x[0] - b[0]);
      if (x[0] > b[1]) r2 += (x[0] - b[1])*(x[0] - b[1]);

      return r2;
    }

    /// Compute squared distance between point and point
    double compute_squared_distance_point(const double* x,
                                          unsigned int node) const
    {
      const double* p = _bbox_coordinates.data() + 2*node;
      return (x[0] - p[0])*(x[0] - p[0]);
    }

    /// Compute bounding box of bounding boxes
    void compute_bbox_of_bboxes(double* bbox,
                                std::size_t& axis,
                                const std::vector<double>& leaf_bboxes,
                                const std::vector<unsigned int>::iterator& begin,
                                const std::vector<unsigned int>::iterator& end)
    {
      typedef std::vector<unsigned int>::const_iterator iterator;

      // Get coordinates for first box
      iterator it = begin;
      const double* b = leaf_bboxes.data() + 2*(*it);
      bbox[0] = b[0];
      bbox[1] = b[1];

      // Compute min and max over remaining boxes
      for (++it; it != end; ++it)
      {
        const double* b = leaf_bboxes.data() + 2*(*it);
        if (b[0] < bbox[0]) bbox[0] = b[0];
        if (b[1] > bbox[1]) bbox[1] = b[1];
      }

      // Compute longest axis
      axis = 0;
    }

    /// Compute bounding box of points
    void compute_bbox_of_points(double* bbox,
                                std::size_t& axis,
                                const std::vector<Point>& points,
                                const std::vector<unsigned int>::iterator& begin,
                                const std::vector<unsigned int>::iterator& end)
    {
      typedef std::vector<unsigned int>::const_iterator iterator;

      // Get coordinates for first point
      iterator it = begin;
      const double* p = points[*it].coordinates();
      bbox[0] = p[0];
      bbox[1] = p[0];

      // Compute min and max over remaining boxes
      for (; it != end; ++it)
      {
        const double* p = points[*it].coordinates();
        if (p[0] < bbox[0]) bbox[0] = p[0];
        if (p[0] > bbox[1]) bbox[1] = p[0];
      }

      // Compute longest axis
      axis = 0;
    }

    /// Sort leaf bounding boxes along given axis
    void sort_bboxes(std::size_t axis,
                     const std::vector<double>& leaf_bboxes,
                     const std::vector<unsigned int>::iterator& begin,
                     const std::vector<unsigned int>::iterator& middle,
                     const std::vector<unsigned int>::iterator& end)
    {
      std::nth_element(begin, middle, end, less_x(leaf_bboxes));
    }

  };

}

#endif
