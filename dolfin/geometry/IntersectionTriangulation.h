// Copyright (C) 2014-2016 Anders Logg and August Johansson
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
// Last changed: 2016-05-08

#include <vector>
#include <dolfin/log/log.h>
#include "CGALExactArithmetic.h"

#ifndef __INTERSECTION_TRIANGULATION_H
#define __INTERSECTION_TRIANGULATION_H

namespace dolfin
{

  // Forward declarations
  class MeshEntity;

  /// This class implements algorithms for computing triangulations of
  /// pairwise intersections of simplices.

  class IntersectionTriangulation
  {
  public:

    //--- High-level intersection triangulation functions ---

    /// Compute triangulation of intersection of two entities.
    ///
    /// *Arguments*
    ///     entity_0 (_MeshEntity_)
    ///         The first entity.
    ///     entity_1 (_MeshEntity_)
    ///         The second entity.
    ///
    /// *Returns*
    ///     std::vector<std::vector<Pointdouble>>
    ///         A vector of simplices
    static std::vector<std::vector<Point>>
    triangulate(const MeshEntity& entity_0,
                const MeshEntity& entity_1);

    /// Compute triangulation of intersection of two entities.
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
    ///     std::vector<std::vector<Point>>
    ///         A vector of simplices
    static std::vector<std::vector<Point>>
    triangulate(const std::vector<Point>& points_0,
                const std::vector<Point>& points_1,
                std::size_t gdim);


    /// Compute triangulation of intersection of a cell with a triangulation
    static std::vector<std::vector<Point>>
    triangulate(const MeshEntity& entity,
                const std::vector<std::vector<Point>>& triangulation,
		std::size_t tdim);

    /// Compute triangulation of intersection of a cell with a triangulation.
    /// This version also handles normals (for boundary triangulation).
    static void
    triangulate(const MeshEntity& entity,
                const std::vector<std::vector<Point>>& triangulation,
                const std::vector<Point>& normals,
                std::vector<std::vector<Point>>& intersection_triangulation,
                std::vector<Point>& intersection_normals,
		std::size_t tdim);

    //--- Low-level intersection triangulation functions ---

    /// Triangulate intersection of segment p0-p1 with segment q0-q1
    static std::vector<Point>
    triangulate_segment_segment(const Point& p0,
                                const Point& p1,
                                const Point& q0,
                                const Point& q1,
                                std::size_t gdim);

    /// Triangulate intersection of segment p0-p1 with segment q0-q1 (1D version)
    static std::vector<Point>
    triangulate_segment_segment_1d(double p0,
				   double p1,
				   double q0,
				   double q1)
    {
      return _triangulate_segment_segment_1d(p0, p1, q0, q1);
    }

    /// Triangulate intersection of segment p0-p1 with segment q0-q1 (2D version)
    static std::vector<Point>
    triangulate_segment_segment_2d(const Point& p0,
				   const Point& p1,
				   const Point& q0,
				   const Point& q1)
    {
      return CHECK_CGAL(_triangulate_segment_segment_2d(p0, p1, q0, q1),
                        cgal_triangulate_segment_segment_2d(p0, p1, q0, q1));
    }

    /// Triangulate intersection of segment p0-p1 with segment q0-q1 (3D version)
    static std::vector<Point>
    triangulate_segment_segment_3d(const Point& p0,
				   const Point& p1,
				   const Point& q0,
				   const Point& q1)
    {
      return _triangulate_segment_segment_3d(p0, p1, q0, q1);
    }

    /// Triangulate intersection of triangle p0-p1-p2 with segment q0-q1
    static std::vector<Point>
    triangulate_triangle_segment(const Point& p0,
                                 const Point& p1,
                                 const Point& p2,
                                 const Point& q0,
                                 const Point& q1,
                                 std::size_t gdim);

    /// Triangulate intersection of triangle p0-p1-p2 with segment q0-q1 (2D version)
    static std::vector<Point>
    triangulate_triangle_segment_2d(const Point& p0,
				    const Point& p1,
				    const Point& p2,
				    const Point& q0,
				    const Point& q1)
    {
      return CHECK_CGAL(_triangulate_triangle_segment_2d(p0, p1, p2, q0, q1),
       			cgal_triangulate_triangle_segment_2d(p0, p1, p2, q0, q1));
    }

    /// Triangulate intersection of triangle p0-p1-p2 with segment q0-q1 (2D version)
    static std::vector<Point>
    triangulate_triangle_segment_3d(const Point& p0,
				    const Point& p1,
				    const Point& p2,
				    const Point& q0,
				    const Point& q1)
    {
      return _triangulate_triangle_segment_3d(p0, p1, p2, q0, q1);
    }

    /// Triangulate intersection of triangle p0-p1-p2 with triangle q0-q1-q2
    static std::vector<std::vector<Point>>
    triangulate_triangle_triangle(const Point& p0,
                                  const Point& p1,
                                  const Point& p2,
                                  const Point& q0,
                                  const Point& q1,
                                  const Point& q2,
				  std::size_t gdim);

    /// Triangulate intersection of triangle p0-p1-p2 with triangle q0-q1-q2 (2D version)
    static std::vector<std::vector<Point>>
    triangulate_triangle_triangle_2d(const Point& p0,
				     const Point& p1,
				     const Point& p2,
				     const Point& q0,
				     const Point& q1,
				     const Point& q2)
    {
      return CHECK_CGAL(_triangulate_triangle_triangle_2d(p0, p1, p2, q0, q1, q2),
      			cgal_triangulate_triangle_triangle_2d(p0, p1, p2, q0, q1, q2));
    }

    /// Triangulate intersection of triangle p0-p1-p2 with triangle q0-q1-q2 (3D version)
    static std::vector<std::vector<Point>>
    triangulate_triangle_triangle_3d(const Point& p0,
				     const Point& p1,
				     const Point& p2,
				     const Point& q0,
				     const Point& q1,
				     const Point& q2)
    {
      return _triangulate_triangle_triangle_3d(p0, p1, p2, q0, q1, q2);
    }

    /// Triangulate intersection of tetrahedron p0-p1-p2-p3 with triangle q0-q1-q2
    static std::vector<std::vector<Point>>
    triangulate_tetrahedron_triangle(const Point& p0,
                                     const Point& p1,
                                     const Point& p2,
                                     const Point& p3,
                                     const Point& q0,
                                     const Point& q1,
                                     const Point& q2)
    {
      return _triangulate_tetrahedron_triangle(p0, p1, p2, p3, q0, q1, q2);
    }

    /// Triangulate intersection of tetrahedron p0-p1-p2-p3 with tetrahedron q0-q1-q2-q3
    static std::vector<std::vector<Point>>
    triangulate_tetrahedron_tetrahedron(const Point& p0,
                                        const Point& p1,
                                        const Point& p2,
                                        const Point& p3,
                                        const Point& q0,
                                        const Point& q1,
                                        const Point& q2,
                                        const Point& q3)
    {
      return _triangulate_tetrahedron_tetrahedron(p0, p1, p2, p3, q0, q1, q2, q3);
    }

  private:

    // Implementation of triangulation functions

    static std::vector<Point>
    _triangulate_segment_segment_1d(double p0,
				    double p1,
				    double q0,
				    double q1);

    static std::vector<Point>
    _triangulate_segment_segment_2d(const Point& p0,
				    const Point& p1,
				    const Point& q0,
				    const Point& q1);

    static std::vector<Point>
    _triangulate_segment_segment_3d(const Point& p0,
				    const Point& p1,
				    const Point& q0,
				    const Point& q1);

    static std::vector<Point>
    _triangulate_triangle_segment_2d(const Point& p0,
				     const Point& p1,
				     const Point& p2,
				     const Point& q0,
				     const Point& q1);

    static std::vector<Point>
    _triangulate_triangle_segment_3d(const Point& p0,
				     const Point& p1,
				     const Point& p2,
				     const Point& q0,
				     const Point& q1);

    static std::vector<std::vector<Point>>
    _triangulate_triangle_triangle_2d(const Point& p0,
				      const Point& p1,
				      const Point& p2,
				      const Point& q0,
				      const Point& q1,
				      const Point& q2);

    static std::vector<std::vector<Point>>
    _triangulate_triangle_triangle_3d(const Point& p0,
				      const Point& p1,
				      const Point& p2,
				      const Point& q0,
				      const Point& q1,
				      const Point& q2);

    static std::vector<std::vector<Point>>
    _triangulate_tetrahedron_triangle(const Point& p0,
				      const Point& p1,
				      const Point& p2,
				      const Point& p3,
				      const Point& q0,
				      const Point& q1,
				      const Point& q2);
    static std::vector<std::vector<Point>>
    _triangulate_tetrahedron_tetrahedron(const Point& p0,
                                         const Point& p1,
                                         const Point& p2,
                                         const Point& p3,
                                         const Point& q0,
                                         const Point& q1,
                                         const Point& q2,
					 const Point& q3);

    //--- Utility functions ---

    // Create triangulation of a convex set of points
    static std::vector<std::vector<Point>>
    graham_scan(const std::vector<Point>& points);
  };
}

#endif
