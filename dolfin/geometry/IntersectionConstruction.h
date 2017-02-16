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
// Last changed: 2017-02-16

#ifndef __INTERSECTION_CONSTRUCTION_H
#define __INTERSECTION_CONSTRUCTION_H

#include <vector>
#include <dolfin/log/log.h>
#include "CGALExactArithmetic.h"
#include "Point.h"

namespace dolfin
{

  // Comparison of points

  struct point_strictly_less
  {
    bool operator()(const dolfin::Point & p0, const dolfin::Point& p1)
    {
      if (p0.x() != p1.x())
        return p0.x() < p1.x();
      return p0.y() < p1.y();
    }
  };

  inline bool operator==(const dolfin::Point& p0, const dolfin::Point& p1)
  {
    return p0.x() == p1.x() and p0.y() == p1.y() and p0.z() == p1.z();
  }

  inline bool operator!=(const dolfin::Point& p0, const dolfin::Point& p1)
  {
    return p0.x() != p1.x() or p0.y() != p1.y() or p0.z() != p1.z();
  }

  inline bool operator<(const dolfin::Point& p0, const dolfin::Point& p1)
  {
    return p0.x() <= p1.x() and p0.y() <= p1.y() and p0.z() <= p1.z();
  }

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

    /* Current status of (re)implementation:

    [ ] intersection_point_point_1d
    [ ] intersection_point_point_2d
    [ ] intersection_point_point_3d
    [ ] intersection_segment_point_1d
    [ ] intersection_segment_point_2d
    [ ] intersection_segment_point_3d
    [ ] intersection_triangle_point_2d
    [ ] intersection_triangle_point_3d
    [ ] intersection_tetrahedron_point_3d
    [ ] intersection_segment_segment_1d
    [ ] intersection_segment_segment_2d
    [ ] intersection_segment_segment_3d
    [ ] intersection_triangle_segment_2d
    [ ] intersection_triangle_segment_3d
    [ ] intersection_tetrahedron_segment_3d
    [ ] intersection_triangle_triangle_2d
    [ ] intersection_triangle_triangle_3d
    [ ] intersection_tetrahedron_triangle_3d
    [ ] intersection_tetrahedron_tetrahedron_3d

    */

    // FIXME: Remove all second-level convenience functions like this one.
    // They are only used internally in the high-level convenience functions.

    // FIXME: Also think about removing the additional wrapper functions
    // since we don't compare with CGAL here anyway.

    // FIXME: Add comment that there are exactly 9 functions and that
    // they are all implemented.

    //--- Point intersections: 9 cases ----//

    /// Compute intersection of points p0 and q0 (1D)
    static std::vector<double>
    intersection_point_point_1d(double p0,
                                double q0);

    /// Compute intersection of points p0 and q0 (2D)
    static std::vector<Point>
    intersection_point_point_2d(const Point& p0,
                                const Point& q0);

    /// Compute intersection of points p0 and q0 (3D)
    static std::vector<Point>
    intersection_point_point_3d(const Point& p0,
                                const Point& q0);

    /// Compute intersection of segment p0-p1 with point q0 (1D)
    static std::vector<double>
    intersection_segment_point_1d(double p0,
                                  double p1,
                                  double q0);

    /// Compute intersection of segment p0-p1 with point q0 (2D)
    static std::vector<Point>
    intersection_segment_point_2d(const Point& p0,
                                  const Point& p1,
                                  const Point& q0);

    /// Compute intersection of segment p0-p1 with point q0 (3D)
    static std::vector<Point>
    intersection_segment_point_3d(const Point& p0,
                                  const Point& p1,
                                  const Point& q0);

    /// Compute intersection of triangle p0-p1-p2 with point q0 (2D)
    static std::vector<Point>
    intersection_triangle_point_2d(const Point& p0,
                                   const Point& p1,
                                   const Point& p2,
                                   const Point& q0);

    /// Compute intersection of triangle p0-p1-p2 with point q0 (3D)
    static std::vector<Point>
    intersection_triangle_point_3d(const Point& p0,
                                   const Point& p1,
                                   const Point& p2,
                                   const Point& q0);

    /// Compute intersection of tetrahedron p0-p1-p2-p3 with point q0 (3D)
    static std::vector<Point>
    intersection_tetrahedron_point_3d(const Point& p0,
                                      const Point& p1,
                                      const Point& p2,
                                      const Point& p3,
                                      const Point& q0);

    //--- Segment intersections: ?? cases ---//

    /// Compute intersection of segment p0-p1 with segment q0-q1 (1D)
    static std::vector<double>
    intersection_segment_segment_1d(double p0,
                                    double p1,
                                    double q0,
                                    double q1);

    /// Compute intersection of segment p0-p1 with segment q0-q1 (2D)
    static std::vector<Point>
    intersection_segment_segment_2d(const Point& p0,
                                    const Point& p1,
                                    const Point& q0,
                                    const Point& q1);

    /// Compute intersection of segment p0-p1 with segment q0-q1 (3D)
    static std::vector<Point>
    intersection_segment_segment_3d(const Point& p0,
                                    const Point& p1,
                                    const Point& q0,
                                    const Point& q1);

    /// Compute intersection of triangle p0-p1-p2 with segment q0-q1 (2D)
    static std::vector<Point>
    intersection_triangle_segment_2d(const Point& p0,
                                     const Point& p1,
                                     const Point& p2,
                                     const Point& q0,
                                     const Point& q1);

    /// Compute intersection of triangle p0-p1-p2 with segment q0-q1 (3D)
    static std::vector<Point>
    intersection_triangle_segment_3d(const Point& p0,
                                     const Point& p1,
                                     const Point& p2,
                                     const Point& q0,
                                     const Point& q1);

    /// Compute intersection of tetrahedron p0-p1-p2-p3 with segment q0-q1 (3D)
    static std::vector<Point>
    intersection_tetrahedron_segment_3d(const Point& p0,
                                        const Point& p1,
                                        const Point& p2,
                                        const Point& p3,
                                        const Point& q0,
                                        const Point& q1);

    // FIXME: The rest of the functions

    /// Compute intersection of triangle p0-p1-p2 with triangle q0-q1-q2 (2D)
    static std::vector<Point>
    intersection_triangle_triangle_2d(const Point& p0,
                                      const Point& p1,
                                      const Point& p2,
                                      const Point& q0,
                                      const Point& q1,
                                      const Point& q2);

    /// Compute intersection of triangle p0-p1-p2 with triangle q0-q1-q2 (3D)
    static std::vector<Point>
    intersection_triangle_triangle_3d(const Point& p0,
                                      const Point& p1,
                                      const Point& p2,
                                      const Point& q0,
                                      const Point& q1,
                                      const Point& q2);

    /// Compute intersection of tetrahedron p0-p1-p2-p3 with triangle q0-q1-q2 (3D)
    static std::vector<Point>
    intersection_tetrahedron_triangle_3d(const Point& p0,
                                         const Point& p1,
                                         const Point& p2,
                                         const Point& p3,
                                         const Point& q0,
                                         const Point& q1,
                                         const Point& q2);

    /// Compute intersection of tetrahedron p0-p1-p2-p3 with tetrahedron q0-q1-q2-q3 (3D)
    static std::vector<Point>
    intersection_tetrahedron_tetrahedron_3d(const Point& p0,
                                            const Point& p1,
                                            const Point& p2,
                                            const Point& p3,
                                            const Point& q0,
                                            const Point& q1,
                                            const Point& q2,
                                            const Point& q3);

  private:

    // Utility functions

    // Add point if equal and mark as added
    static inline void add_if_equal(std::vector<double>& points,
                                    double p,
                                    double q,
                                    bool& pi,
                                    bool& qi)
    {
      if (!pi and p == q)
      {
        points.push_back(p);
        pi = qi = true;
      }
    }

    // Add point if equal and mark as added
    static inline void add_if_equal(std::vector<Point>& points,
                                    const Point& p,
                                    const Point& q,
                                    bool& pi,
                                    bool& qi)
    {
      if (!pi and p == q)
      {
        points.push_back(p);
        pi = qi = true;
      }
    }

    // Add points to vector
    static inline void add(std::vector<Point>& points,
                           const std::vector<Point>& _points)
    {
      points.insert(points.end(), _points.begin(), _points.end());
    }

    // Strictly unique points using == operator
    // TODO: Will the points be unique most of the times? Should this function
    // filter out inplace?
    static std::vector<Point> _unique_points(const std::vector<Point>& points);

    // Determinant of 3 x 3 matrix
    static double _det(const Point& ab,
                       const Point& dc,
                       const Point& ec);


    static std::vector<double>
    _intersection_segment_segment_1d_old(double p0,
                                         double p1,
                                         double q0,
                                         double q1);

    static std::vector<Point>
    _intersection_segment_segment_2d_old(const Point& p0,
                                         const Point& p1,
                                         const Point& q0,
                                         const Point& q1);

    static std::vector<Point>
    _intersection_triangle_segment_2d_old(const Point& p0,
                                          const Point& p1,
                                          const Point& p2,
                                          const Point& q0,
                                          const Point& q1);

    static std::vector<Point>
    _intersection_triangle_triangle_2d_old(const Point& p0,
                                           const Point& p1,
                                           const Point& p2,
                                           const Point& q0,
                                           const Point& q1,
                                           const Point& q2);

    std::vector<Point>
    _intersection_tetrahedron_triangle_3d_old(const Point& p0,
                                              const Point& p1,
                                              const Point& p2,
                                              const Point& p3,
                                              const Point& q0,
                                              const Point& q1,
                                              const Point& q2);

    static std::vector<Point>
    _intersection_tetrahedron_tetrahedron_3d_old(const Point& p0,
                                                 const Point& p1,
                                                 const Point& p2,
                                                 const Point& p3,
                                                 const Point& q0,
                                                 const Point& q1,
                                                 const Point& q2,
                                                 const Point& q3);

  };

}

#endif
