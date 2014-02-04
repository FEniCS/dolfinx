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

#ifndef __COLLISION_DETECTION_H
#define __COLLISION_DETECTION_H

namespace dolfin
{

  // Forward declarations
  class MeshEntity;

  /// This class implements algorithms for detecting pairwise
  /// collisions between mesh entities of varying dimensions.

  class CollisionDetection
  {
  public:

    /// Check whether edge collides with edge.
    ///
    /// *Arguments*
    ///     a (_Point_)
    ///         The first point of the first edge.
    ///     b (_Point_)
    ///         The second point of the first edge.
    ///     c (_Point_)
    ///         The first point of the second edge.
    ///     d (_Point_)
    ///         The second point of the second edge.
    ///
    /// *Returns*
    ///     bool
    ///         True iff objects collide.
    static bool collides_edge_edge(const Point& a,
				   const Point& b,
				   const Point& c,
				   const Point& d);

    /// Check whether triangle collides with point.
    ///
    /// *Arguments*
    ///     triangle (_MeshEntity_)
    ///         The triangle.
    ///     point (_Point_)
    ///         The point.
    ///
    /// *Returns*
    ///     bool
    ///         True iff objects collide.
    static bool collides_triangle_point(const MeshEntity& triangle,
					const Point& point);

    /// Check whether triangle collides with triangle.
    ///
    /// *Arguments*
    ///     triangle_0 (_MeshEntity_)
    ///         The first triangle.
    ///     triangle_1 (_MeshEntity_)
    ///         The second triangle.
    ///
    /// *Returns*
    ///     bool
    ///         True iff objects collide.
    static bool collides_triangle_triangle(const MeshEntity& triangle_0,
					   const MeshEntity& triangle_1);

    /// Check whether tetrahedron collides with point.
    ///
    /// *Arguments*
    ///     tetrahedron (_MeshEntity_)
    ///         The tetrahedron.
    ///     point (_Point_)
    ///         The point.
    ///
    /// *Returns*
    ///     bool
    ///         True iff objects collide.
    static bool collides_tetrahedron_point(const MeshEntity& tetrahedron,
                                           const Point& point);

    /// Check whether tetrahedron collides with triangle.
    ///
    /// *Arguments*
    ///     tetrahedron (_MeshEntity_)
    ///         The tetrahedron.
    ///     triangle (_MeshEntity_)
    ///         The triangle.
    ///
    /// *Returns*
    ///     bool
    ///         True iff objects collide.
    static bool collides_tetrahedron_triangle(const MeshEntity& tetrahedron,
                                              const MeshEntity& triangle);

    /// Check whether tetrahedron collides with tetrahedron.
    ///
    /// *Arguments*
    ///     tetrahedron_0 (_MeshEntity_)
    ///         The first tetrahedron.
    ///     tetrahedron_1 (_MeshEntity_)
    ///         The second tetrahedron.
    ///
    /// *Returns*
    ///     bool
    ///         True iff objects collide.
    static bool collides_tetrahedron_tetrahedron(const MeshEntity& tetrahedron_0,
                                                 const MeshEntity& tetrahedron_1);

  private:

    // Helper function for collides_triangle_triangle
    static bool compute_intervals(const Point& N1,
				  const Point& V0,
				  const Point& V1,
				  const Point& V2,
				  const Point& U0,
				  const Point& U1,
				  const Point& U2,
				  const Point& VV,
				  const Point& D,
				  double D0D1,
				  double D0D2,
				  Point& ABC,
				  double& X0,
				  double& X1);

    // Helper function for collides_triangle_triangle
    static bool coplanar_triangle_triangle(const Point& N,
					   const Point& V0,
					   const Point& V1,
					   const Point& V2,
					   const Point& U0,
					   const Point& U1,
					   const Point& U2);
    // Helper function for collides_triangle_triangle
    static bool edge_against_tri_edges(int i0,
				       int i1,
				       const Point& V0,
				      const Point& V1,
				      const Point& U0,
				      const Point& U1,
				      const Point& U2);

    // Helper
    static bool edge_edge_test(int i0,
			       int i1,
			       double Ax,
			       double Ay,
			       const Point& V0,
			       const Point& U0,
			       const Point& U1);

    // Helper
    static bool point_in_tri(int i0,
			     int i1,
			     const Point& V0,
			     const Point& U0,
			     const Point& U1,
			     const Point& U2);
    

    // Helper function for collides_tetrahedron_tetrahedron: checks if plane pv1 is a separating plane. Stores local coordinates bc and the mask bit maskEdges.
    static bool separating_plane_face_A_1(const std::vector<Point>& pv1,
				   const Point& n,
				   std::vector<double>& bc,
				   int& maskEdges);

    // Helper function for collides_tetrahedron_tetrahedron: checks if plane v1,v2 is a separating plane. Stores local coordinates bc and the mask bit maskEdges.
    static bool separating_plane_face_A_2(const std::vector<Point>& v1,
				   const std::vector<Point>& v2,
				   const Point& n,
				   std::vector<double>& bc,
				   int& maskEdges);
		
    // Helper function for collides_tetrahedron_tetrahedron: checks if plane pv2 is a separating plane.
    static bool separating_plane_face_B_1(const std::vector<Point>& P_V2,
				   const Point& n) 
    {
      return ((P_V2[0].dot(n) > 0) &&
	      (P_V2[1].dot(n) > 0) &&
	      (P_V2[2].dot(n) > 0) &&
	      (P_V2[3].dot(n) > 0));
    }

    // Helper function for collides_tetrahedron_tetrahedron: checks if plane v1,v2 is a separating plane.  
    static bool separating_plane_face_B_2(const std::vector<Point>& V1,
				   const std::vector<Point>& V2,
				   const Point& n) 
    {
      return (((V1[0]-V2[1]).dot(n) > 0) &&
	      ((V1[1]-V2[1]).dot(n) > 0) &&
	      ((V1[2]-V2[1]).dot(n) > 0) &&
	      ((V1[3]-V2[1]).dot(n) > 0));
    }
    
    // Helper function for collides_tetrahedron_tetrahedron: checks if edge is in the plane separating faces f0 and f1. 
    static bool separating_plane_edge_A(const std::vector<std::vector<double> >& Coord_1,
				 const std::vector<int>& masks,
				 int f0, 
				 int f1);

  };

}

#endif
