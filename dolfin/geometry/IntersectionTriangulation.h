// Copyright (C) 2014 Anders Logg and August Johansson
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
// Last changed: 2014-05-28

#include <vector>
#include <dolfin/log/log.h>

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

    /// Compute triangulation of intersection of two entities
    ///
    /// *Arguments*
    ///     entity_0 (_MeshEntity_)
    ///         The first entity.
    ///     entity_1 (_MeshEntity_)
    ///         The second entity.
    ///
    /// *Returns*
    ///     std::vector<double>
    ///         A flattened array of simplices of dimension
    ///         num_simplices x num_vertices x gdim =
    ///         num_simplices x (tdim + 1) x gdim
    static std::vector<double>
    triangulate_intersection(const MeshEntity& entity_0,
                             const MeshEntity& entity_1);

    /// Compute triangulation of intersection of two intervals
    ///
    /// *Arguments*
    ///     T0 (_MeshEntity_)
    ///         The first interval.
    ///     T1 (_MeshEntity_)
    ///         The second interval.
    ///
    /// *Returns*
    ///     std::vector<double>
    ///         A flattened array of simplices of dimension
    ///         num_simplices x num_vertices x gdim =
    ///         num_simplices x (tdim + 1) x gdim
    static std::vector<double>
    triangulate_intersection_interval_interval(const MeshEntity& interval_0,
                                               const MeshEntity& interval_1);

    /// Compute triangulation of intersection of a triangle and an interval
    ///
    /// *Arguments*
    ///     T0 (_MeshEntity_)
    ///         The triangle.
    ///     T1 (_MeshEntity_)
    ///         The interval.
    ///
    /// *Returns*
    ///     std::vector<double>
    ///         A flattened array of simplices of dimension
    ///         num_simplices x num_vertices x gdim =
    ///         num_simplices x (tdim + 1) x gdim
    static std::vector<double>
    triangulate_intersection_triangle_interval(const MeshEntity& triangle,
                                               const MeshEntity& interval);

    /// Compute triangulation of intersection of two triangles
    ///
    /// *Arguments*
    ///     T0 (_MeshEntity_)
    ///         The first triangle.
    ///     T1 (_MeshEntity_)
    ///         The second triangle.
    ///
    /// *Returns*
    ///     std::vector<double>
    ///         A flattened array of simplices of dimension
    ///         num_simplices x num_vertices x gdim =
    ///         num_simplices x (tdim + 1) x gdim
    static std::vector<double>
    triangulate_intersection_triangle_triangle(const MeshEntity& triangle_0,
                                               const MeshEntity& triangle_1);

    /// Compute triangulation of intersection of a tetrahedron and a triangle
    ///
    /// *Arguments*
    ///     T0 (_MeshEntity_)
    ///         The tetrahedron.
    ///     T1 (_MeshEntity_)
    ///         The triangle
    ///
    /// *Returns*
    ///     std::vector<double>
    ///         A flattened array of simplices of dimension
    ///         num_simplices x num_vertices x gdim =
    ///         num_simplices x (tdim + 1) x gdim
    static std::vector<double>
    triangulate_intersection_tetrahedron_triangle(const MeshEntity& tetrahedron,
                                                  const MeshEntity& triangle);

    /// Compute triangulation of intersection of two tetrahedra
    ///
    /// *Arguments*
    ///     T0 (_MeshEntity_)
    ///         The first tetrahedron.
    ///     T1 (_MeshEntity_)
    ///         The second tetrahedron.
    ///
    /// *Returns*
    ///     std::vector<double>
    ///         A flattened array of simplices of dimension
    ///         num_simplices x num_vertices x gdim =
    ///         num_simplices x (tdim + 1) x gdim
    static std::vector<double>
    triangulate_intersection_tetrahedron_tetrahedron(const MeshEntity& tetrahedron_0,
                                                     const MeshEntity& tetrahedron_1);

    // Function for general intersection computation of two simplices
    // with different topological dimension but the same geometrical
    // dimension
    static std::vector<double>
    triangulate_intersection(const std::vector<Point>& s0,
                             std::size_t tdim0,
                             const std::vector<Point>& s1,
                             std::size_t tdim1,
                             std::size_t gdim);

    // Function for computing the intersection of a cell with a flat
    // vector of simplices with topological dimension tdim. The
    // geometrical dimension is assumed to be the same as for the
    // cell.
    static std::vector<double>
    triangulate_intersection(const MeshEntity& cell,
                             const std::vector<double> &triangulation,
                             std::size_t tdim);

    // Function for computing the intersection of a cell with a flat
    // vector of simplices with topological dimension tdim. The
    // geometrical dimension is assumed to be the same as for the
    // cell. The corresponding normals are also saved.
    static void
    triangulate_intersection(const MeshEntity& cell,
                             const std::vector<double>& triangulation,
                             const std::vector<Point>& normals,
                             std::vector<double>& intersection_triangulation,
                             std::vector<Point>& intersection_normals,
                             std::size_t tdim);

    // Function for computing the intersection of two triangles given
    // by std::vector<Point>.
    static std::vector<double>
    triangulate_intersection_triangle_triangle(const std::vector<Point>& tri_0,
                                               const std::vector<Point>& tri_1);

    // FIXME: this shouldn't be public.
    // Function for creating the convex triangulation of a set of points
    static std::vector<double>
      graham_scan(const std::vector<Point>& points);

  private:

    // Function for computing the intersection of two intervals given
    // by std::vector<Point>.
    static std::vector<double>
    triangulate_intersection_interval_interval(const std::vector<Point>& interval_0,
                                               const std::vector<Point>& interval_1,
                                               std::size_t gdim);

    // Function for computing the intersection of a triangle and an interval
    // by std::vector<Point>.
    static std::vector<double>
    triangulate_intersection_triangle_interval(const std::vector<Point>& triangle,
                                               const std::vector<Point>& interval,
                                               std::size_t gdim);

    // Function for computing the intersection of two tetrahedra given
    // by std::vector<Point>.
    static std::vector<double>
    triangulate_intersection_tetrahedron_tetrahedron(const std::vector<Point>& tet_0,
                                                     const std::vector<Point>& tet_1);

    // Function for computing the intersection of a tetrahedron with a
    // triangle given by std::vector<Point>.
    static std::vector<double>
    triangulate_intersection_tetrahedron_triangle(const std::vector<Point>& tet,
                                                  const std::vector<Point>& tri);


    // Helper function
    // FIXME
    static bool intersection_edge_edge_2d(const Point& a,
				       const Point& b,
				       const Point& c,
				       const Point& d,
				       Point& pt);
    static bool intersection_edge_edge(const Point& a,
				       const Point& b,
				       const Point& c,
				       const Point& d,
				       Point& pt);

    // Helper function
    static bool intersection_face_edge(const Point& r,
				       const Point& s,
				       const Point& t,
				       const Point& a,
				       const Point& b,
				       Point& pt);

    static double minimum_angle(double* a, double* b, double* c);

  };

}

#endif
