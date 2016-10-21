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
// Modified by August Johansson 2016
//
// First added:  2013-05-02
// Last changed: 2016-07-05

#ifndef __BOUNDING_BOX_TREE_2D_H
#define __BOUNDING_BOX_TREE_2D_H

#include <algorithm>
#include <vector>
#include <dolfin/common/constants.h>
#include "GenericBoundingBoxTree.h"
#include <dolfin/geometry/predicates.h>

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

    // Return geometric dimension
    std::size_t gdim() const { return 2; }

    // Return bounding box coordinates for node
    const double* get_bbox_coordinates(unsigned int node) const
    {
      return _bbox_coordinates.data() + 4*node;
    }

    // Check whether point (x) is in bounding box (node)
    bool point_in_bbox(const double* x, unsigned int node) const
    {
      const double* b = _bbox_coordinates.data() + 4*node;
      const double eps0 = DOLFIN_EPS_LARGE*(b[2] - b[0]);
      const double eps1 = DOLFIN_EPS_LARGE*(b[3] - b[1]);
      return (b[0] - eps0 <= x[0] && x[0] <= b[2] + eps0 &&
              b[1] - eps1 <= x[1] && x[1] <= b[3] + eps1);

      // // FIXME: It is possible to also use Shewchuk robust geometry
      // // here as well, just as in bbox_in_bbox
      // const double* b = _bbox_coordinates.data() + 4*node;
      // double p0[2] = {b[0], b[1]};
      // double p1[2] = {b[2], b[1]};
      // if (orient2d(const_cast<double*>(x), p0, p1) > 0)
      // 	return false;

      // double p2[2] = {b[0], b[3]};
      // if (orient2d(const_cast<double*>(x), p0, p2) > 0)
      // 	return false;

      // double p3[2] = {b[2], b[3]};
      // if (orient2d(const_cast<double*>(x), p2, p3) > 0)
      // 	return false;

      // if (orient2d(const_cast<double*>(x), p3, p1) > 0)
      // 	return false;

      // return true;
    }

    // Check whether bounding box (a) collides with bounding box (node)
    bool bbox_in_bbox(const double* a, unsigned int node) const
    {
      // FIXME: Some testing suggests that we should use Shewchuk's
      // routines for collision testing
      // const double* b = _bbox_coordinates.data() + 4*node;
      // const double eps0 = DOLFIN_EPS*(b[2] - b[0]);
      // const double eps1 = DOLFIN_EPS*(b[3] - b[1]);
      // return (b[0] - eps0 <= a[2] && a[0] <= b[2] + eps0 &&
      // 	      b[1] - eps1 <= a[3] && a[1] <= b[3] + eps1);

      const double* b = _bbox_coordinates.data() + 4*node;

      double ll[2] = {a[2], a[3]};
      double p0[2] = {b[0], b[1]};
      double p1[2] = {b[2], b[1]};
      if (orient2d(ll, p0, p1) < 0)
      	return false;

      double p2[2] = {b[0], b[3]};
      if (orient2d(ll, p2, p0) < 0)
      	return false;

      double ur[2] = {a[0], a[1]};
      double p3[2] = {b[2], b[3]};
      if (orient2d(ur, p3, p2) < 0)
      	return false;

      if (orient2d(ur, p1, p3) < 0)
      	return false;

      return true;
    }

    // Compute squared distance between point and bounding box
    double compute_squared_distance_bbox(const double* x,
                                         unsigned int node) const
    {
      // Note: Some else-if might be in order here but I assume the
      // compiler can do a better job at optimizing/parallelizing this
      // version. This is also the way the algorithm is presented in
      // Ericsson.

      const double* b = _bbox_coordinates.data() + 4*node;
      double r2 = 0.0;

      if (x[0] < b[0]) r2 += (x[0] - b[0])*(x[0] - b[0]);
      if (x[0] > b[2]) r2 += (x[0] - b[2])*(x[0] - b[2]);
      if (x[1] < b[1]) r2 += (x[1] - b[1])*(x[1] - b[1]);
      if (x[1] > b[3]) r2 += (x[1] - b[3])*(x[1] - b[3]);

      return r2;
    }

    // Compute squared distance between point and point
    double compute_squared_distance_point(const double* x,
                                          unsigned int node) const
    {
      const double* p = _bbox_coordinates.data() + 4*node;
      return (x[0] - p[0])*(x[0] - p[0]) + (x[1] - p[1])*(x[1] - p[1]);
    }

    // Compute bounding box of bounding boxes
    void compute_bbox_of_bboxes(double* bbox,
                                std::size_t& axis,
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
      for (++it; it != end; ++it)
      {
        const double* b = leaf_bboxes.data() + 4*(*it);
        if (b[0] < bbox[0]) bbox[0] = b[0];
        if (b[1] < bbox[1]) bbox[1] = b[1];
        if (b[2] > bbox[2]) bbox[2] = b[2];
        if (b[3] > bbox[3]) bbox[3] = b[3];
      }

      // Compute longest axis
      const double x = bbox[2] - bbox[0];
      const double y = bbox[3] - bbox[1];

      if (x > y)
        axis = 0;
      else
        axis = 1;
    }

    // Compute bounding box of points
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
      bbox[1] = p[1];
      bbox[2] = p[0];
      bbox[3] = p[1];

      // Compute min and max over remaining points
      for (; it != end; ++it)
      {
        const double* p = points[*it].coordinates();
        if (p[0] < bbox[0]) bbox[0] = p[0];
        if (p[1] < bbox[1]) bbox[1] = p[1];
        if (p[0] > bbox[2]) bbox[2] = p[0];
        if (p[1] > bbox[3]) bbox[3] = p[1];
      }

      // Compute longest axis
      const double x = bbox[2] - bbox[0];
      const double y = bbox[3] - bbox[1];

      if (x > y)
        axis = 0;
      else
        axis = 1;
    }

    // Sort leaf bounding boxes along given axis
    void sort_bboxes(std::size_t axis,
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
