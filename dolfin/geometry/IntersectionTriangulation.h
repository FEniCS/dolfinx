// Copyright (C) 2014 Anders Logg
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
// Last changed: 2014-02-03

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

  private:

    // Helper function
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
  };

}

#endif
