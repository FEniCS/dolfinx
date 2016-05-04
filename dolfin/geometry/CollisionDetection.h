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
// Last changed: 2016-05-04

#include <vector>
#include <dolfin/log/log.h>
#include "CGALExactArithmetic.h"

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

    //--- High-level collision detection predicates ---

    /// Check whether entity collides with point.
    ///
    /// *Arguments*
    ///     entity (_MeshEntity_)
    ///         The entity.
    ///     point (_Point_)
    ///         The point.
    ///
    /// *Returns*
    ///     bool
    ///         True iff entity collides with cell.
    static bool collides(const MeshEntity& entity,
			 const Point& point);

    /// Check whether two entities collide.
    ///
    /// *Arguments*
    ///     entity_0 (_MeshEntity_)
    ///         The first entity.
    ///     entity_1 (_MeshEntity_)
    ///         The second entity.
    ///
    /// *Returns*
    ///     bool
    ///         True iff entity collides with cell.
    static bool collides(const MeshEntity& entity_0,
			 const MeshEntity& entity_1);

    //--- Low-level collision detection predicates ---

    /// Check whether segment p0-p1 collides with point
    static bool collides_segment_point(const Point& p0, const Point& p1,
				       const Point& point)
    {
      return CHECK_CGAL(_collides_segment_point(p0, p1, point),
                        cgal_collides_segment_point(p0, p1, point));
    }

    /// Check whether segment p0-p1 collides with segment q0-q1
    static bool collides_segment_segment(const Point& p0, const Point& p1,
					 const Point& q0, const Point& q1)
    {
      return CHECK_CGAL(_collides_segment_segment(p0, p1, q0, q1),
			cgal_collides_segment_segment(p0, p1, q0, q1));
    }

    /// Check whether segment p0-p1 collides with segment q0-q1 (1D version)
    static bool collides_segment_segment_1d(double p0,
                                            double p1,
                                            double q0,
                                            double q1)
    {
      return _collides_segment_segment_1d(p0, p1, q0, q1);
    }

    /// Check whether triangle p0-p1-p2 collides with point
    static bool collides_triangle_point(const Point& p0,
					const Point& p1,
					const Point& p2,
					const Point& point)
    {
      return CHECK_CGAL(_collides_triangle_point(p0, p1, p2, point),
                        cgal_collides_triangle_point(p0, p1, p2, point));
    }

    /// Check whether triangle p0-p1-p2 collides with point (2D version)
    static bool collides_triangle_point_2d(const Point& p0,
                                           const Point& p1,
                                           const Point& p2,
                                           const Point& point)
    {
      return CHECK_CGAL(_collides_triangle_point_2d(p0, p1, p2, point),
                        cgal_collides_triangle_point_2d(p0, p1, p2, point));
    }

    /// Check whether triangle p0-p1-p2 collides with segment q0-q1
    static bool collides_triangle_segment(const Point& p0,
					  const Point& p1,
					  const Point& p2,
					  const Point& q0,
					  const Point& q1)
    {
      return CHECK_CGAL(_collides_triangle_segment(p0, p1, p2, q0, q1),
                        cgal_collides_triangle_segment(p0, p1, p2, q0, q1));
    }

    /// Check whether triangle p0-p1-p2 collides with triangle q0-q1-q2
    static bool collides_triangle_triangle(const Point& p0,
					   const Point& p1,
					   const Point& p2,
					   const Point& q0,
					   const Point& q1,
					   const Point& q2)
    {
      return CHECK_CGAL(_collides_triangle_triangle(p0, p1, p2, q0, q1, q2),
                        cgal_collides_triangle_triangle(p0, p1, p2, q0, q1, q2));
    }

    /// Check whether tetrahedron p0-p1-p2-p3 collides with point
    static bool collides_tetrahedron_point(const Point& p0,
					   const Point& p1,
					   const Point& p2,
					   const Point& p3,
					   const Point& point)
    {
      return _collides_tetrahedron_point(p0, p1, p2, p3, point);
    }

    /// Check whether tetrahedron p0-p1-p2-p3 collides with triangle q0-q1-q2
    static bool collides_tetrahedron_triangle(const Point& p0,
					      const Point& p1,
					      const Point& p2,
					      const Point& p3,
					      const Point& q0,
					      const Point& q1,
					      const Point& q2)
    {
      return _collides_tetrahedron_triangle(p0, p1, p2, p3,
					    q0, q1, q2);
    }

    /// Check whether tetrahedron p0-p1-p2-p3 collides with tetrahedron q0-q1-q2
    static bool collides_tetrahedron_tetrahedron(const Point& p0,
						 const Point& p1,
						 const Point& p2,
						 const Point& p3,
						 const Point& q0,
						 const Point& q1,
						 const Point& q2,
						 const Point& q3)
    {
      return _collides_tetrahedron_tetrahedron(p0, p1, p2, p3,
					       q0, q1, q2, q3);
    }

  private:

    // Implementation of collision detection predicates

    static bool _collides_segment_point(const Point& p0,
					const Point& p1,
					const Point& point);

    static bool _collides_segment_segment(const Point& p0,
                                          const Point& p1,
					  const Point& q0,
                                          const Point& q1);

    static bool _collides_segment_segment_1d(double p0,
                                             double p1,
                                             double q0,
                                             double q1);

    static bool _collides_triangle_point(const Point& p0,
					 const Point& p1,
					 const Point& p2,
					 const Point& point);

    static bool _collides_triangle_point_2d(const Point& p0,
					    const Point& p1,
					    const Point& p2,
					    const Point& point);

    static bool _collides_triangle_segment(const Point& p0,
					   const Point& p1,
					   const Point& p2,
					   const Point& q0,
					   const Point& q1);

    static bool _collides_triangle_triangle(const Point& p0,
					    const Point& p1,
					    const Point& p2,
					    const Point& q0,
					    const Point& q1,
					    const Point& q2);

    static bool _collides_tetrahedron_point(const Point& p0,
					    const Point& p1,
					    const Point& p2,
					    const Point& p3,
					    const Point& point);

    static bool _collides_tetrahedron_triangle(const Point& p0,
					       const Point& p1,
					       const Point& p2,
					       const Point& p3,
					       const Point& q0,
					       const Point& q1,
					       const Point& q2);

    static bool _collides_tetrahedron_tetrahedron(const Point& p0,
						  const Point& p1,
						  const Point& p2,
						  const Point& p3,
						  const Point& q0,
						  const Point& q1,
						  const Point& q2,
						  const Point& q3);

    //--- Utility functions ---

    // Utility function for triangle-triangle collision
    static bool edge_edge_test(int i0,
                               int i1,
                               double Ax,
                               double Ay,
			       const Point& V0,
			       const Point& U0,
			       const Point& U1);

    // Utility function for triangle-triangle collision
    static bool edge_against_tri_edges(int i0,
                                       int i1,
				       const Point& V0,
				       const Point& V1,
				       const Point& U0,
				       const Point& U1,
				       const Point& U2);

    // Utility function for triangle-triangle collision
    static bool point_in_triangle(int i0,
                                  int i1,
                                  const Point& V0,
                                  const Point& U0,
                                  const Point& U1,
                                  const Point& U2);

    // Utility function for triangle-triangle collision
    static bool coplanar_tri_tri(const Point& N,
				 const Point& V0,
				 const Point& V1,
				 const Point& V2,
				 const Point& U0,
				 const Point& U1,
				 const Point& U2);

    // Utility function for triangle-triangle collision
    static bool compute_intervals(double VV0,
                                  double VV1,
                                  double VV2,
				  double D0,
                                  double D1,
                                  double D2,
				  double D0D1,
                                  double D0D2,
				  double& A,
                                  double& B,
                                  double& C,
				  double& X0,
                                  double& X1);

    // Utility function for collides_tetrahedron_tetrahedron: checks if
    // plane pv1 is a separating plane. Stores local coordinates bc
    // and the mask bit mask_edges.
    static bool separating_plane_face_A_1(const std::vector<Point>& pv1,
					  const Point& n,
					  std::vector<double>& bc,
					  int& mask_edges);

    // Utility function for collides_tetrahedron_tetrahedron: checks if
    // plane v1, v2 is a separating plane. Stores local coordinates bc
    // and the mask bit mask_edges.
    static bool separating_plane_face_A_2(const std::vector<Point>& v1,
					  const std::vector<Point>& v2,
					  const Point& n,
					  std::vector<double>& bc,
					  int& mask_edges);

    // Utility function for collides_tetrahedron_tetrahedron: checks if
    // plane pv2 is a separating plane.
    static bool separating_plane_face_B_1(const std::vector<Point>& P_V2,
					  const Point& n)
    {
      return ((P_V2[0].dot(n) > 0) &&
	      (P_V2[1].dot(n) > 0) &&
	      (P_V2[2].dot(n) > 0) &&
	      (P_V2[3].dot(n) > 0));
    }

    // Utility function for collides_tetrahedron_tetrahedron: checks if
    // plane v1, v2 is a separating plane.
    static bool separating_plane_face_B_2(const std::vector<Point>& V1,
					  const std::vector<Point>& V2,
					  const Point& n)
    {
      return (((V1[0] - V2[1]).dot(n) > 0) &&
	      ((V1[1] - V2[1]).dot(n) > 0) &&
	      ((V1[2] - V2[1]).dot(n) > 0) &&
	      ((V1[3] - V2[1]).dot(n) > 0));
    }

    // Utility function for collides_tetrahedron_tetrahedron: checks if
    // edge is in the plane separating faces f0 and f1.
    static bool separating_plane_edge_A(const std::vector<std::vector<double> >& coord_1,
					const std::vector<int>& masks,
					int f0,
					int f1);

  };

}

#endif
