// Copyright (C) 2014-2016 Anders Logg, August Johansson and Benjamin Kehlet
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
// First added:  2014-02-03
// Last changed: 2017-02-10

#ifndef __INTERSECTION_CONSTRUCTION_H
#define __INTERSECTION_CONSTRUCTION_H

#include <vector>
#include <dolfin/log/log.h>
#include "CGALExactArithmetic.h"
#include "Point.h"

namespace dolfin
{

  // Forward declarations
  class MeshEntity;

  /// This class implements algorithms for computing pairwise
  /// intersections of simplices. The computed intersection is always
  /// convex and represented as a set of points s.t. the intersection
  /// is the convex hull of these points.

  class IntersectionConstruction
  {
  public:

    /// Compute intersection of two entities.
    ///
    /// *Arguments*
    ///     entity_0 (_MeshEntity_)
    ///         The first entity.
    ///     entity_1 (_MeshEntity_)
    ///         The second entity.
    ///
    /// *Returns*
    ///     std::vector<Pointdouble>
    ///         A vector of points s.t. the intersection is the convex hull of
    ///         these points.
    static std::vector<Point>
    intersection(const MeshEntity& entity_0,
                 const MeshEntity& entity_1);

    /// Compute intersection of two entities.
    /// This version takes two vectors of points representing the entities.
    ///
    /// *Arguments*
    ///     points_0 (std::vector<Point>)
    ///         The vertex coordinates of the first entity.
    ///     points_1 (std::vector<Point>)
    ///         The vertex coordinates of the second entity.
    ///     gdim (std::size_t)
    ///         The geometric dimension.
    ///
    /// *Returns*
    ///     std::vector<Point>
    ///         A vector of points s.t. the intersection is the convex hull of
    ///         these points.
    static std::vector<Point>
    intersection(const std::vector<Point>& points_0,
                 const std::vector<Point>& points_1,
                 std::size_t gdim);

    //--- Low-level intersection construction functions ---

    /// Compute intersection of segment p0-p1 with segment q0-q1
    static std::vector<Point>
    intersection_segment_segment(const Point& p0,
                                 const Point& p1,
                                 const Point& q0,
                                 const Point& q1,
                                 std::size_t gdim);

    /// Compute intersection of segment p0-p1 with segment q0-q1 (1D version)
    static std::vector<double>
    intersection_segment_segment_1d(double p0,
                                    double p1,
                                    double q0,
                                    double q1)
    {
      return _intersection_segment_segment_1d(p0, p1, q0, q1);
    }

    /// Compute intersection of segment p0-p1 with segment q0-q1 (2D version)
    static std::vector<Point>
    intersection_segment_segment_2d(const Point& p0,
                                    const Point& p1,
                                    const Point& q0,
                                    const Point& q1)
    {
      return _intersection_segment_segment_2d(p0, p1, q0, q1);
    }

    /// Compute intersection of segment p0-p1 with segment q0-q1 (3D version)
    static std::vector<Point>
    intersection_segment_segment_3d(const Point& p0,
                                    const Point& p1,
                                    const Point& q0,
                                    const Point& q1)
    {
      return _intersection_segment_segment_3d(p0, p1, q0, q1);
    }

    /// Compute intersection of triangle p0-p1-p2 with segment q0-q1
    static std::vector<Point>
    intersection_triangle_segment(const Point& p0,
                                  const Point& p1,
                                  const Point& p2,
                                  const Point& q0,
                                  const Point& q1,
                                  std::size_t gdim);

    /// Compute intersection of triangle p0-p1-p2 with segment q0-q1 (2D version)
    static std::vector<Point>
    intersection_triangle_segment_2d(const Point& p0,
                                     const Point& p1,
                                     const Point& p2,
                                     const Point& q0,
                                     const Point& q1)
    {
      return _intersection_triangle_segment_2d(p0, p1, p2, q0, q1);
    }

    /// Compute intersection of triangle p0-p1-p2 with segment q0-q1 (2D version)
    static std::vector<Point>
    intersection_triangle_segment_3d(Point p0,
                                     Point p1,
                                     Point p2,
                                     Point q0,
                                     Point q1)
    {
      return _intersection_triangle_segment_3d(p0, p1, p2, q0, q1);
    }

    /// Compute intersection of triangle p0-p1-p2 with triangle q0-q1-q2
    static std::vector<Point>
    intersection_triangle_triangle(const Point& p0,
                                   const Point& p1,
                                   const Point& p2,
                                   const Point& q0,
                                   const Point& q1,
                                   const Point& q2,
                                   std::size_t gdim);

    /// Compute intersection of triangle p0-p1-p2 with triangle q0-q1-q2 (2D version)
    static std::vector<Point>
    intersection_triangle_triangle_2d(const Point& p0,
                                      const Point& p1,
                                      const Point& p2,
                                      const Point& q0,
                                      const Point& q1,
                                      const Point& q2)
    {
      return _intersection_triangle_triangle_2d(p0, p1, p2, q0, q1, q2);
    }

    /// Compute intersection of triangle p0-p1-p2 with triangle q0-q1-q2 (3D version)
    static std::vector<Point>
    intersection_triangle_triangle_3d(const Point& p0,
                                      const Point& p1,
                                      const Point& p2,
                                      const Point& q0,
                                      const Point& q1,
                                      const Point& q2)
    {
      return _intersection_triangle_triangle_3d(p0, p1, p2, q0, q1, q2);
    }

    /// Compute intersection of tetrahedron p0-p1-p2-p3 with triangle q0-q1-q2
    static std::vector<Point>
    intersection_tetrahedron_triangle(const Point& p0,
                                      const Point& p1,
                                      const Point& p2,
                                      const Point& p3,
                                      const Point& q0,
                                      const Point& q1,
                                      const Point& q2)
    {
      return _intersection_tetrahedron_triangle_3d(p0, p1, p2, p3, q0, q1, q2);
    }

    /// Compute intersection of tetrahedron p0-p1-p2-p3 with tetrahedron q0-q1-q2-q3
    static std::vector<Point>
    intersection_tetrahedron_tetrahedron(const Point& p0,
                                         const Point& p1,
                                         const Point& p2,
                                         const Point& p3,
                                         const Point& q0,
                                         const Point& q1,
                                         const Point& q2,
                                         const Point& q3)
    {
      return _intersection_tetrahedron_tetrahedron_3d(p0, p1, p2, p3, q0, q1, q2, q3);
    }

  private:

    // Implementation of intersection construction functions

    static std::vector<double>
    _intersection_segment_segment_1d(double p0,
                                     double p1,
                                     double q0,
                                     double q1);

    static std::vector<Point>
    _intersection_segment_segment_2d(Point p0,
                                     Point p1,
                                     Point q0,
                                     Point q1);

    static std::vector<Point>
    _intersection_segment_segment_3d(const Point& p0,
                                     const Point& p1,
                                     const Point& q0,
                                     const Point& q1);

    static std::vector<Point>
    _intersection_triangle_segment_2d(const Point& p0,
                                      const Point& p1,
                                      const Point& p2,
                                      const Point& q0,
                                      const Point& q1);

    static std::vector<Point>
    _intersection_triangle_segment_3d(Point p0,
                                      Point p1,
                                      Point p2,
                                      Point q0,
                                      Point q1);

    static std::vector<Point>
    _intersection_triangle_triangle_2d(Point p0,
                                       Point p1,
                                       Point p2,
                                       Point q0,
                                       Point q1,
                                       Point q2);

    static std::vector<Point>
    _intersection_triangle_triangle_3d(const Point& p0,
                                       const Point& p1,
                                       const Point& p2,
                                       const Point& q0,
                                       const Point& q1,
                                       const Point& q2);

    static std::vector<Point>
    _intersection_tetrahedron_triangle_3d(const Point& p0,
                                          const Point& p1,
                                          const Point& p2,
                                          const Point& p3,
                                          const Point& q0,
                                          const Point& q1,
                                          const Point& q2);
    static std::vector<Point>
    _intersection_tetrahedron_tetrahedron_3d(const Point& p0,
                                             const Point& p1,
                                             const Point& p2,
                                             const Point& p3,
                                             const Point& q0,
                                             const Point& q1,
                                             const Point& q2,
                                             const Point& q3);

    // Utility functions

    // FIXME: Think about passing points by reference

    // Strictly unique points using == operator
    static std::vector<Point> _unique_points(std::vector<Point> points);

    // Determinant of 3 x 3 matrix
    static double _det(Point ab, Point dc, Point ec);

    // Numerically robust cross product
    static Point _cross_product(Point a, Point b, Point c);

    // Major (largest) axis of vector
    static std::size_t _major_axis(const Point& v);

    // Project point to major axis plane
    static Point _project_point(const Point& p, std::size_t major_axis);

  };

}

#endif
