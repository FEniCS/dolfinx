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
// Modified by Chris Richardson, 2014.
//
// First added:  2014-02-03
// Last changed: 2016-05-29
//
//-----------------------------------------------------------------------------
// Special note regarding the function collides_tetrahedron_tetrahedron
//-----------------------------------------------------------------------------
//
// The source code for the tetrahedron-tetrahedron collision test is
// from Fabio Ganovelli, Federico Ponchio and Claudio Rocchini: Fast
// Tetrahedron-Tetrahedron Overlap Algorithm, Journal of Graphics
// Tools, 7(2), 2002, and is under the following copyright:
//
// Visual Computing Group
// IEI Institute, CNUCE Institute, CNR Pisa
//
// Copyright(C) 2002 by Fabio Ganovelli, Federico Ponchio and Claudio
// Rocchini
//
// All rights reserved.
//
// Permission to use, copy, modify, distribute and sell this software
// and its documentation for any purpose is hereby granted without
// fee, provided that the above copyright notice appear in all copies
// and that both that copyright notice and this permission notice
// appear in supporting documentation. the author makes no
// representations about the suitability of this software for any
// purpose. It is provided "as is" without express or implied
// warranty.
//
//-----------------------------------------------------------------------------
// Special note regarding the function collides_triangle_triangle
//-----------------------------------------------------------------------------
//
// The source code for the triangle-triangle collision test is from
// Tomas Moller: A Fast Triangle-Triangle Intersection Test, Journal
// of Graphics Tools, 2(2), 1997, and is in the public domain.
//
//-----------------------------------------------------------------------------

#include <dolfin/mesh/MeshEntity.h>
#include "predicates.h"
#include "Point.h"
#include "GeometryDebugging.h"
#include "CollisionDetection.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
// High-level collision detection predicates
//-----------------------------------------------------------------------------
bool CollisionDetection::collides(const MeshEntity& entity,
                                  const Point& point)
{
  // Get data
  const MeshGeometry& g = entity.mesh().geometry();
  const unsigned int* v = entity.entities(0);
  const std::size_t tdim = entity.mesh().topology().dim();
  const std::size_t gdim = entity.mesh().geometry().dim();

  // Pick correct specialized implementation
  if (tdim == 1)
    return collides_segment_point(g.point(v[0]), g.point(v[1]), point);

  if (tdim == 2 && gdim == 2)
    return collides_triangle_point_2d(g.point(v[0]),
                                      g.point(v[1]),
                                      g.point(v[2]),
                                      point);

  if (tdim == 2 && gdim == 3)
    return collides_triangle_point_3d(g.point(v[0]),
				      g.point(v[1]),
				      g.point(v[2]),
				      point);

  if (tdim == 3)
    return collides_tetrahedron_point(g.point(v[0]),
				      g.point(v[1]),
				      g.point(v[2]),
				      g.point(v[3]),
				      point);

  dolfin_error("CollisionDetection.cpp",
               "compute entity-point collision",
               "Not implemented for dimensions %d / %d", tdim, gdim);

  return false;
}
//-----------------------------------------------------------------------------
bool CollisionDetection::collides(const MeshEntity& entity_0,
				  const MeshEntity& entity_1)
{
  // Get data
  const MeshGeometry& g0 = entity_0.mesh().geometry();
  const MeshGeometry& g1 = entity_1.mesh().geometry();
  const unsigned int* v0 = entity_0.entities(0);
  const unsigned int* v1 = entity_1.entities(0);
  const std::size_t d0 = entity_0.dim();
  const std::size_t d1 = entity_1.dim();
  const std::size_t gdim = g0.dim();
  dolfin_assert(gdim == g1.dim());

  // Pick correct specialized implementation
  if (d0 == 1 && d1 == 1)
    return collides_segment_segment(g0.point(v0[0])[0],
				    g0.point(v0[1])[0],
				    g1.point(v1[0])[0],
				    g1.point(v1[1])[0],
				    gdim);

  if (d0 == 1 && d1 == 2)
    return collides_triangle_segment(g1.point(v1[0]),
				     g1.point(v1[1]),
				     g1.point(v1[2]),
				     g0.point(v0[0]),
				     g0.point(v0[1]),
				     gdim);

  if (d0 == 2 && d1 == 1)
    return collides_triangle_segment(g0.point(v0[0]),
				     g0.point(v0[1]),
				     g0.point(v0[2]),
				     g1.point(v1[0]),
				     g1.point(v1[1]),
				     gdim);

  if (d0 == 2 && d1 == 2)
    return collides_triangle_triangle(g0.point(v0[0]),
				      g0.point(v0[1]),
				      g0.point(v0[2]),
				      g1.point(v1[0]),
				      g1.point(v1[1]),
				      g1.point(v1[2]),
				      gdim);

  if (d0 == 2 && d1 == 3)
    return collides_tetrahedron_triangle(g1.point(v1[0]),
                                         g1.point(v1[1]),
                                         g1.point(v1[2]),
                                         g1.point(v1[3]),
                                         g0.point(v0[0]),
                                         g0.point(v0[1]),
                                         g0.point(v0[2]));

  if (d0 == 3 && d1 == 2)
    return collides_tetrahedron_triangle(g0.point(v0[0]),
                                         g0.point(v0[1]),
                                         g0.point(v0[2]),
                                         g0.point(v0[3]),
                                         g1.point(v1[0]),
                                         g1.point(v1[1]),
                                         g1.point(v1[2]));

  if (d0 == 3 && d1 == 3)
    return collides_tetrahedron_tetrahedron(g0.point(v0[0]),
                                            g0.point(v0[1]),
                                            g0.point(v0[2]),
                                            g0.point(v0[3]),
                                            g1.point(v1[0]),
                                            g1.point(v1[1]),
                                            g1.point(v1[2]),
                                            g1.point(v1[3]));

  dolfin_error("CollisionDetection.cpp",
               "compute entity-entity collision",
               "Not implemented for topological dimensions %d / %d and geometrical dimension %d", d0, d1, gdim);

  return false;
}
//-----------------------------------------------------------------------------
// Low-level collision detection predicates
//-----------------------------------------------------------------------------
bool CollisionDetection::collides_segment_segment(const Point& p0,
						  const Point& p1,
						  const Point& q0,
						  const Point& q1,
						  std::size_t gdim)
{
  switch (gdim)
  {
  case 1:
    return collides_segment_segment_1d(p0[0], p1[0], q0[0], q1[0]);
  case 2:
    return collides_segment_segment_2d(p0, p1, q0, q1);
  case 3:
    return collides_segment_segment_3d(p0, p1, q0, q1);
  default:
    dolfin_error("CollisionDetection.cpp",
		 "compute segment-segment collision ",
		 "Unknown dimension (Implemented for dimension 1, 2 and 3)");
  }

  return false;
}
//------------------------------------------------------------------------------
bool CollisionDetection::collides_triangle_point(const Point& p0,
						 const Point& p1,
						 const Point& p2,
						 const Point& point,
						 std::size_t gdim)
{
  switch (gdim)
  {
  case 2:
    return collides_triangle_point_2d(p0, p1, p2, point);
  case 3:
    return collides_triangle_point_3d(p0, p1, p2, point);
  default:
    dolfin_error("CollisionDetection.cpp",
		 "compute triangle-point collision ",
		 "Implemented only for dimension 2 and 3.");
  }
  return false;
}
//------------------------------------------------------------------------------
bool CollisionDetection::collides_triangle_segment(const Point& p0,
						   const Point& p1,
						   const Point& p2,
						   const Point& q0,
						   const Point& q1,
						   std::size_t gdim)
{
  switch (gdim)
  {
  case 2:
    return collides_triangle_segment_2d(p0, p1, p2, q0, q1);
  case 3:
    return collides_triangle_segment_3d(p0, p1, p2, q0, q1);
  default:
    dolfin_error("CollisionDetection.cpp",
		 "compute triangle-segment collision ",
		 "Implmented only for dimension 2 and 3.");
  }
  return false;
}
//------------------------------------------------------------------------------
bool CollisionDetection::collides_triangle_triangle(const Point& p0,
						    const Point& p1,
						    const Point& p2,
						    const Point& q0,
						    const Point& q1,
						    const Point& q2,
						    std::size_t gdim)
{
  switch (gdim)
  {
  case 2:
    return collides_triangle_triangle_2d(p0, p1, p2, q0, q1, q2);
  case 3:
    return collides_triangle_triangle_3d(p0, p1, p2, q0, q1, q2);
  default:
    dolfin_error("CollisionDetection.cpp",
		 "compute triangle-triangle collision ",
		 "Implmented only for dimension 2 and 3.");
  }
  return false;
}
//-----------------------------------------------------------------------------
bool CollisionDetection::_collides_segment_point(Point p0,
						 Point p1,
						 Point point)
{
  const double orientation = orient2d(p0.coordinates(),
                                      p1.coordinates(),
                                      point.coordinates());

  return orientation == 0 &&
    (point-p0).squared_norm() <= (p1-p0).squared_norm() &&
    (point-p1).squared_norm() <= (p0-p1).squared_norm();
}
//------------------------------------------------------------------------------
bool CollisionDetection::_collides_segment_segment_1d(double p0,
                                                      double p1,
                                                      double q0,
                                                      double q1)
{
  // Get range
  const double a0 = std::min(p0, p1);
  const double b0 = std::max(p0, p1);
  const double a1 = std::min(q0, q1);
  const double b1 = std::max(q0, q1);

  // Check for collisions
  const double dx = std::min(b0 - a0, b1 - a1);
  const double eps = std::max(DOLFIN_EPS_LARGE, DOLFIN_EPS_LARGE*dx);
  const bool result = b1 > a0 - eps && a1 < b0 + eps;

  return result;
}
//-----------------------------------------------------------------------------
bool operator==(const Point& a, const Point& b)
{
  return a.x() == b.x() && a.y() == b.y() && a.z() == b.z();
}

bool CollisionDetection::_collides_segment_segment_2d(Point p0,
						      Point p1,
						      Point q0,
						      Point q1)
{
  //std::cout << __FUNCTION__ << std::endl;

  // Vertex vertex collision
  if (p0 == q0 || p0 == q1 || p1 == q0 || p1 == q1)
    return true;

  const double q0_q1_p0 = orient2d(q0.coordinates(),
                                   q1.coordinates(),
                                   p0.coordinates());
  const double q0_q1_p1 = orient2d(q0.coordinates(),
                                   q1.coordinates(),
                                   p1.coordinates());
  const double p0_p1_q0 = orient2d(p0.coordinates(),
                                   p1.coordinates(),
                                   q0.coordinates());
  const double p0_p1_q1 = orient2d(p0.coordinates(),
                                   p1.coordinates(),
                                   q1.coordinates());

  // Vertex edge (interior) collision

  if (q0_q1_p0 == 0 && (p0-q0).squared_norm() < (q1-q0).squared_norm() && (p0-q1).squared_norm() < (q0-q1).squared_norm())
    return true;
  if (q0_q1_p1 == 0 && (p1-q0).squared_norm() < (q1-q0).squared_norm() && (p1-q1).squared_norm() < (q0-q1).squared_norm())
    return true;
  if (p0_p1_q0 == 0 && (q0-p0).squared_norm() < (p1-p0).squared_norm() && (q0-p1).squared_norm() < (p0-p1).squared_norm())
    return true;
  if (p0_p1_q1 == 0 && (q1-p0).squared_norm() < (p1-p0).squared_norm() && (q1-p1).squared_norm() < (p0-p1).squared_norm())
    return true;

  // std::cout << std::signbit(q0_q1_p0)<< ' '<< std::signbit(q0_q1_p1)<<' '<< std::signbit(p0_p1_q0) <<' '<< std::signbit(p0_p1_q1) <<std::endl;

  return q0_q1_p0*q0_q1_p1 < 0 && p0_p1_q0*p0_p1_q1 < 0;

}
//-----------------------------------------------------------------------------
bool CollisionDetection::_collides_triangle_point_2d(Point p0,
						     Point p1,
						     Point p2,
						     Point point)
{
  // Check triangle orientation
  const int sign = std::signbit(orient2d(p0.coordinates(),
					 p1.coordinates(),
					 p2.coordinates()));

  // std::cout << tools::drawtriangle({{p0,p1,p2}})<<tools::drawtriangle({{point}})<<std::endl;

  // std::cout << sign <<' '<<std::signbit(orient2d(p0.coordinates(), p1.coordinates(), point.coordinates()))<<' '<<std::signbit(orient2d(p1.coordinates(), p2.coordinates(), point.coordinates()))<<' '<<std::signbit(orient2d(p2.coordinates(), p0.coordinates(), point.coordinates()))<<std::endl;

  // The point is inside if all triangles formed have the same orientation
  if (sign == std::signbit(orient2d(p0.coordinates(), p1.coordinates(), point.coordinates())) and
      sign == std::signbit(orient2d(p1.coordinates(), p2.coordinates(), point.coordinates())) and
      sign == std::signbit(orient2d(p2.coordinates(), p0.coordinates(), point.coordinates())))
    return true;
  else
    return false;
}
//-----------------------------------------------------------------------------
bool CollisionDetection::_collides_triangle_point_3d(const Point& p0,
						     const Point& p1,
						     const Point& p2,
						     const Point& point)
{
  // Algorithm from http://www.blackpawn.com/texts/pointinpoly/

  // Vectors defining each edge in consistent orientation
  const Point r0 = p0 - p2;
  const Point r1 = p1 - p0;
  const Point r2 = p2 - p1;

  // Normal to triangle: should be the same as
  // r2.cross(r1) and r0.cross(r2).
  Point normal = r1.cross(r0);

  Point r = point - p0;
  // Check point is in plane of triangle (for manifold)
  double volume = r.dot(normal);
  if (std::abs(volume) > DOLFIN_EPS)
    return false;

  // Compute normal to triangle based on point and first edge
  // Dot product of two normals should be positive, if inside.
  Point pnormal = r.cross(r0);
  double t1 = normal.dot(pnormal);
  if (t1 < 0) return false;

  // Repeat for each edge
  r = point - p1;
  pnormal = r.cross(r1);
  double t2 = normal.dot(pnormal);
  if (t2 < 0) return false;

  r = point - p2;
  pnormal = r.cross(r2);
  double t3 = normal.dot(pnormal);
  if (t3 < 0) return false;

  return true;
}
//-----------------------------------------------------------------------------
bool CollisionDetection::_collides_triangle_segment_2d(const Point& p0,
						       const Point& p1,
						       const Point& p2,
						       const Point& q0,
						       const Point& q1)
{
  // Check if end points are in triangle
  if (collides_triangle_point_2d(p0, p1, p2, q0))
    return true;
  if (collides_triangle_point_2d(p0, p1, p2, q1))
    return true;

  // Check if any of the triangle edges are cut by the segment
  if (collides_segment_segment_2d(p0, p1, q0, q1))
    return true;
  if (collides_segment_segment_2d(p0, p2, q0, q1))
    return true;
  if (collides_segment_segment_2d(p1, p2, q0, q1))
    return true;

  return false;
}
//-----------------------------------------------------------------------------
bool CollisionDetection::_collides_triangle_segment_3d(const Point& p0,
						       const Point& p1,
						       const Point& p2,
						       const Point& q0,
						       const Point& q1)
{
  // Check if end points are in triangle
  if (collides_triangle_point_3d(p0, p1, p2, q0))
    return true;
  if (collides_triangle_point_3d(p0, p1, p2, q1))
    return true;

  // Check if any of the triangle edges are cut by the segment
  if (collides_segment_segment_3d(p0, p1, q0, q1))
    return true;
  if (collides_segment_segment_3d(p0, p2, q0, q1))
    return true;
  if (collides_segment_segment_3d(p1, p2, q0, q1))
    return true;

  return false;
}
//------------------------------------------------------------------------------
bool CollisionDetection::_collides_triangle_triangle_2d(const Point& p0,
							const Point& p1,
							const Point& p2,
							const Point& q0,
							const Point& q1,
							const Point& q2)
{
  // Pack points as vectors
  std::vector<Point> tri_0({p0, p1, p2});
  std::vector<Point> tri_1({q0, q1, q2});

  // Extract coordinates
  double t0[3][2] = {{p0[0], p0[1]}, {p1[0], p1[1]}, {p2[0], p2[1]}};
  double t1[3][2] = {{q0[0], q0[1]}, {q1[0], q1[1]}, {q2[0], q2[1]}};

  // Find all vertex-cell collisions
  const int s0 = std::signbit(orient2d(t0[0], t0[1], t0[2])) == true ? -1 : 1;
  const int s1 = std::signbit(orient2d(t1[0], t1[1], t1[2])) == true ? -1 : 1;

  for (std::size_t i = 0; i < 3; ++i)
  {
    if (s1*orient2d(t1[0], t1[1], t0[i]) >= 0. and
  	s1*orient2d(t1[1], t1[2], t0[i]) >= 0. and
  	s1*orient2d(t1[2], t1[0], t0[i]) >= 0.)
      return true;

    if (s0*orient2d(t0[0], t0[1], t1[i]) >= 0. and
  	s0*orient2d(t0[1], t0[2], t1[i]) >= 0. and
  	s0*orient2d(t0[2], t0[0], t1[i]) >= 0.)
      return true;
  }

  // Find all edge-edge collisions
  for (std::size_t i0 = 0; i0 < 3; i0++)
  {
    const std::size_t j0 = (i0 + 1) % 3;
    const Point& p0 = tri_0[i0];
    const Point& q0 = tri_0[j0];
    for (std::size_t i1 = 0; i1 < 3; i1++)
    {
      const std::size_t j1 = (i1 + 1) % 3;
      const Point& p1 = tri_1[i1];
      const Point& q1 = tri_1[j1];
      if (CollisionDetection::collides_segment_segment_2d(p0, q0, p1, q1))
	return true;
    }
  }

  return false;
}
//-----------------------------------------------------------------------------
bool CollisionDetection::_collides_tetrahedron_point(const Point& p0,
						     const Point& p1,
						     const Point& p2,
						     const Point& p3,
						     const Point& point)
{
  // Algorithm from http://www.blackpawn.com/texts/pointinpoly/
  // See also "Real-Time Collision Detection" by Christer Ericson.

  const Point *p[4] = {&p0, &p1, &p2, &p3};

  // Consider each face in turn
  for (unsigned int i = 0; i != 4; ++i)
  {
    // Compute vectors relative to p[i]
    const Point v1 = *p[(i + 1)%4] - *p[i];
    const Point v2 = *p[(i + 2)%4] - *p[i];
    const Point v3 = *p[(i + 3)%4] - *p[i];
    const Point v = point - *p[i];
    // Normal to plane containing v1 and v2
    const Point n1 = v1.cross(v2);
    // Find which side of face plane points v and v3 lie
    const double t1 = n1.dot(v);
    const double t2 = n1.dot(v3);
    // Catch case where point is exactly on plane
    // otherwise require points to be on same side
    if (t1 != 0.0 and std::signbit(t1) != std::signbit(t2))
      return false;
  }
  return true;
}
//-----------------------------------------------------------------------------
bool CollisionDetection::_collides_tetrahedron_triangle(const Point& p0,
							const Point& p1,
							const Point& p2,
							const Point& p3,
							const Point& q0,
							const Point& q1,
							const Point& q2)
{
  // Collision is determined by first if any triangle vertex is inside
  // the tetrahedron. If not, we continue checking the intersection of
  // the triangle with the four faces of the tetrahedron.

  // Triangle vertex in tetrahedron collision
  if (collides_tetrahedron_point(p0, p1, p2, p3, q0))
    return true;
  if (collides_tetrahedron_point(p0, p1, p2, p3, q1))
    return true;
  if (collides_tetrahedron_point(p0, p1, p2, p3, q2))
    return true;

  // Triangle-triangle collision tests
  if (collides_triangle_triangle_3d(q0, q1, q2, p1, p2, p3))
    return true;
  if (collides_triangle_triangle_3d(q0, q1, q2, p0, p2, p3))
    return true;
  if (collides_triangle_triangle_3d(q0, q1, q2, p0, p1, p3))
    return true;
  if (collides_triangle_triangle_3d(q0, q1, q2, p0, p1, p2))
    return true;

  return false;
}
//-----------------------------------------------------------------------------
bool CollisionDetection::_collides_tetrahedron_tetrahedron(const Point& p0,
							   const Point& p1,
							   const Point& p2,
							   const Point& p3,
							   const Point& q0,
							   const Point& q1,
							   const Point& q2,
							   const Point& q3)
{
  // This algorithm checks whether two tetrahedra intersect.

  // Algorithm and source code from Fabio Ganovelli, Federico Ponchio
  // and Claudio Rocchini: Fast Tetrahedron-Tetrahedron Overlap
  // Algorithm, Journal of Graphics Tools, 7(2), 2002. DOI:
  // 10.1080/10867651.2002.10487557. Source code available at
  // http://web.archive.org/web/20031130075955/
  // http://www.acm.org/jgt/papers/GanovelliPonchioRocchini02/tet_a_tet.html

  const std::vector<Point> V1 = {{ p0, p1, p2, p3 }};
  const std::vector<Point> V2 = {{ q0, q1, q2, q3 }};

  // Get the vectors between V2 and V1[0]
  std::vector<Point> P_V1(4);
  for (std::size_t i = 0; i < 4; ++i)
    P_V1[i] = V2[i]-V1[0];

  // Data structure for edges of V1 and V2
  std::vector<Point> e_v1(5), e_v2(5);
  e_v1[0] = V1[1] - V1[0];
  e_v1[1] = V1[2] - V1[0];
  e_v1[2] = V1[3] - V1[0];
  Point n = e_v1[1].cross(e_v1[0]);

  // Maybe flip normal. Normal should be outward.
  if (n.dot(e_v1[2]) > 0)
    n *= -1;
  std::vector<int> masks(4);
  std::vector<std::vector<double>> Coord_1(4, std::vector<double>(4));
  if (separating_plane_face_A_1(P_V1, n, Coord_1[0], masks[0]))
    return false;
  n = e_v1[0].cross(e_v1[2]);

  // Maybe flip normal
  if (n.dot(e_v1[1]) > 0)
    n *= -1;
  if (separating_plane_face_A_1(P_V1, n, Coord_1[1], masks[1]))
    return false;
  if (separating_plane_edge_A(Coord_1, masks, 0, 1))
    return false;
  n = e_v1[2].cross(e_v1[1]);

  // Maybe flip normal
  if (n.dot(e_v1[0]) > 0)
    n *= -1;
  if (separating_plane_face_A_1(P_V1, n, Coord_1[2], masks[2]))
    return false;
  if (separating_plane_edge_A(Coord_1, masks, 0, 2))
    return false;
  if (separating_plane_edge_A(Coord_1, masks, 1,2))
    return false;
  e_v1[4] = V1[3] - V1[1];
  e_v1[3] = V1[2] - V1[1];
  n = e_v1[3].cross(e_v1[4]);

  // Maybe flip normal. Note the < since e_v1[0]=v1-v0.
  if (n.dot(e_v1[0]) < 0)
    n *= -1;
  if (separating_plane_face_A_2(V1, V2, n, Coord_1[3], masks[3]))
    return false;
  if (separating_plane_edge_A(Coord_1, masks, 0, 3))
    return false;
  if (separating_plane_edge_A(Coord_1, masks, 1, 3))
    return false;
  if (separating_plane_edge_A(Coord_1, masks, 2, 3))
    return false;
  if ((masks[0] | masks[1] | masks[2] | masks[3] )!= 15)
    return true;

  // From now on, if there is a separating plane, it is parallel to a
  // face of b.
  std::vector<Point> P_V2(4);
  for (std::size_t i = 0; i < 4; ++i)
    P_V2[i] = V1[i] - V2[0];
  e_v2[0] = V2[1] - V2[0];
  e_v2[1] = V2[2] - V2[0];
  e_v2[2] = V2[3] - V2[0];
  n = e_v2[1].cross(e_v2[0]);

  // Maybe flip normal
  if (n.dot(e_v2[2]) > 0)
    n *= -1;
  if (separating_plane_face_B_1(P_V2, n))
    return false;
  n=e_v2[0].cross(e_v2[2]);

  // Maybe flip normal
  if (n.dot(e_v2[1]) > 0)
    n *= -1;
  if (separating_plane_face_B_1(P_V2, n))
    return false;
  n = e_v2[2].cross(e_v2[1]);

  // Maybe flip normal
  if (n.dot(e_v2[0]) > 0)
    n *= -1;
  if (separating_plane_face_B_1(P_V2, n))
    return false;
  e_v2[4] = V2[3] - V2[1];
  e_v2[3] = V2[2] - V2[1];
  n = e_v2[3].cross(e_v2[4]);

  // Maybe flip normal. Note the < since e_v2[0] = V2[1] - V2[0].
  if (n.dot(e_v2[0]) < 0)
    n *= -1;
  if (separating_plane_face_B_2(V1, V2, n))
    return false;

  return true;
}
//-----------------------------------------------------------------------------
bool CollisionDetection::edge_edge_test(int i0,
                                        int i1,
                                        double Ax,
                                        double Ay,
					const Point& V0,
					const Point& U0,
					const Point& U1)
{
  // Helper function for triangle triangle collision. Test edge vs
  // edge.

  // Here we have the option of classifying adjacent edges of two
  // triangles as colliding by changing > to >= and < to <= below.

  const double Bx = U0[i0] - U1[i0];
  const double By = U0[i1] - U1[i1];
  const double Cx = V0[i0] - U0[i0];
  const double Cy = V0[i1] - U0[i1];
  const double f = Ay*Bx - Ax*By;
  const double d = By*Cx - Bx*Cy;

  if ((f > 0 && d >= 0 && d <= f) ||
      (f < 0 && d <= 0 && d >= f))
  {
    const double e = Ax*Cy - Ay*Cx;
    if (f > 0)
    {
      // Allow or not allow adjacent edges as colliding:
      //if (e >= 0 && e <= f) return true;
      if (e > 0 && e < f)
        return true;
    }
    else
    {
      // Allow or not allow adjacent edges as colliding:
      //if (e <= 0 && e >= f) return true;
      if (e < 0 && e > f)
        return true;
    }
  }
  return false;
}
//-----------------------------------------------------------------------------
bool CollisionDetection::edge_against_tri_edges(int i0,
                                                int i1,
						const Point& V0,
						const Point& V1,
						const Point& U0,
						const Point& U1,
						const Point& U2)
{
  // Helper function for triangle triangle collision
  const double Ax = V1[i0] - V0[i0];
  const double Ay = V1[i1] - V0[i1];

  // Test edge U0,U1 against V0,V1
  if (edge_edge_test(i0, i1, Ax, Ay, V0, U0, U1))
    return true;

  // Test edge U1,U2 against V0,V1
  if (edge_edge_test(i0, i1, Ax, Ay, V0, U1, U2))
    return true;

  // Test edge U2,U1 against V0,V1
  if (edge_edge_test(i0, i1, Ax, Ay, V0, U2, U0))
    return true;

  return false;
}
//-----------------------------------------------------------------------------
bool CollisionDetection::point_in_triangle(int i0,
                                           int i1,
                                           const Point& V0,
                                           const Point& U0,
                                           const Point& U1,
                                           const Point& U2)
{
  // Helper function for triangle triangle collision
  // Is T1 completely inside T2?
  // Check if V0 is inside triangle(U0, U1, U2)
  double a = U1[i1] - U0[i1];
  double b = -(U1[i0] - U0[i0]);
  double c = -a*U0[i0] - b*U0[i1];
  const double d0 = a*V0[i0] + b*V0[i1] + c;

  a = U2[i1] - U1[i1];
  b = -(U2[i0] - U1[i0]);
  c = -a*U1[i0] - b*U1[i1];
  const double d1 = a*V0[i0] + b*V0[i1] + c;

  a = U0[i1] - U2[i1];
  b = -(U0[i0] - U2[i0]);
  c = -a*U2[i0] - b*U2[i1];
  const double d2 = a*V0[i0] + b*V0[i1] + c;

  if (d0*d1 > 0. && d0*d2 > 0.)
    return true;

  return false;
}
//-----------------------------------------------------------------------------
bool CollisionDetection::coplanar_tri_tri(const Point& N,
					  const Point& V0,
					  const Point& V1,
					  const Point& V2,
					  const Point& U0,
					  const Point& U1,
					  const Point& U2)
{
  // Helper function for triangle triangle collision

  double A[3];
  int i0,i1;

  // First project onto an axis-aligned plane, that maximizes the area
  // of the triangles, compute indices: i0,i1.
  A[0] = std::abs(N[0]);
  A[1] = std::abs(N[1]);
  A[2] = std::abs(N[2]);

  if (A[0] > A[1])
  {
    if (A[0] > A[2])
    {
      i0 = 1; // A[0] is greatest
      i1 = 2;
    }
    else
    {
      i0 = 0; // A[2] is greatest
      i1 = 1;
    }
  }
  else // A[0] <= A[1]
  {
    if (A[2] > A[1])
    {
      i0 = 0; // A[2] is greatest
      i1 = 1;
    }
    else
    {
      i0 = 0; // A[1] is greatest
      i1 = 2;
    }
  }

  // Test all edges of triangle 1 against the edges of triangle 2
  if (edge_against_tri_edges(i0, i1, V0, V1, U0, U1, U2))
    return true;
  if (edge_against_tri_edges(i0, i1, V1, V2, U0, U1, U2))
    return true;
  if (edge_against_tri_edges(i0, i1, V2, V0, U0, U1, U2))
    return true;

  // Finally, test if tri1 is totally contained in tri2 or vice versa
  if (point_in_triangle(i0, i1, V0, U0, U1, U2))
    return true;
  if (point_in_triangle(i0, i1, U0, V0, V1, V2))
    return true;

  return false;
}
//-----------------------------------------------------------------------------
bool CollisionDetection::compute_intervals(double VV0,
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
					   double& X1)
{
  // Helper function for triangle triangle collision

  if (D0D1 > 0.)
  {
    // Here we know that D0D2<=0.0, that is D0, D1 are on the same
    // side, D2 on the other or on the plane
    A = VV2;
    B = (VV0 - VV2)*D2;
    C = (VV1 - VV2)*D2;
    X0 = D2 - D0;
    X1 = D2 - D1;
  }
  else if (D0D2 > 0.)
  {
    // Here we know that d0d1<=0.0
    A = VV1;
    B = (VV0 - VV1)*D1;
    C = (VV2 - VV1)*D1;
    X0 = D1 - D0;
    X1 = D1 - D2;
  }
  else if (D1*D2 > 0. || D0 != 0.)
  {
    // Here we know that d0d1<=0.0 or that D0!=0.0
    A = VV0;
    B = (VV1 - VV0)*D0;
    C = (VV2 - VV0)*D0;
    X0 = D0 - D1;
    X1 = D0 - D2;
  }
  else if (D1 != 0.)
  {
    A = VV1;
    B = (VV0 - VV1)*D1;
    C = (VV2 - VV1)*D1;
    X0 = D1 - D0;
    X1 = D1 - D2;
  }
  else if (D2 != 0.)
  {
    A = VV2;
    B = (VV0 - VV2)*D2;
    C = (VV1 - VV2)*D2;
    X0 = D2 - D0;
    X1 = D2 - D1;
  }
  else {
    // Go to coplanar test
    return true;
  }

  return false;
}
//-----------------------------------------------------------------------------
bool
CollisionDetection::separating_plane_face_A_1(const std::vector<Point>& pv1,
					      const Point& n,
					      std::vector<double>& coord,
					      int&  mask_edges)
{
  // Helper function for tetrahedron-tetrahedron collision test:
  // checks if plane pv1 is a separating plane. Stores local
  // coordinates and the mask bit mask_edges.

  mask_edges = 0;
  const int shifts[4] = {1, 2, 4, 8};

  for (std::size_t i = 0; i < 4; ++i)
  {
    coord[i] = pv1[i].dot(n);
    if (coord[i] > 0)
      mask_edges |= shifts[i];
  }

  return (mask_edges == 15);
}
//-----------------------------------------------------------------------------
bool
CollisionDetection::separating_plane_face_A_2(const std::vector<Point>& V1,
					      const std::vector<Point>& V2,
					      const Point& n,
					      std::vector<double>& coord,
					      int&  mask_edges)
{
  // Helper function for tetrahedron-tetrahedron collision test:
  // checks if plane v1,v2 is a separating plane. Stores local
  // coordinates and the mask bit mask_edges.

  mask_edges = 0;
  const int shifts[4] = {1, 2, 4, 8};

  for (std::size_t i = 0; i < 4; ++i)
  {
    coord[i] = (V2[i] - V1[1]).dot(n);
    if (coord[i] > 0)
      mask_edges |= shifts[i];
  }

  return (mask_edges == 15);
}
//-----------------------------------------------------------------------------
bool CollisionDetection::separating_plane_edge_A(const std::vector<std::vector<double>>& coord_1,
						 const std::vector<int>& masks, int f0, int f1)
{
  // Helper function for tetrahedron-tetrahedron collision: checks if
  // edge is in the plane separating faces f0 and f1.

  const std::vector<double>& coord_f0 = coord_1[f0];
  const std::vector<double>& coord_f1 = coord_1[f1];

  int maskf0 = masks[f0];
  int maskf1 = masks[f1];

  if ((maskf0 | maskf1) != 15) // if there is a vertex of b
    return false; // included in (-,-) return false

  maskf0 &= (maskf0 ^ maskf1); // exclude the vertices in (+,+)
  maskf1 &= (maskf0 ^ maskf1);

  // edge 0: 0--1
  if ((maskf0 & 1) && // the vertex 0 of b is in (-,+)
      (maskf1 & 2)) // the vertex 1 of b is in (+,-)
    if ((coord_f0[1]*coord_f1[0] - coord_f0[0]*coord_f1[1]) > 0)
      // the edge of b (0,1) intersect (-,-) (see the paper)
      return false;

  if ((maskf0 & 2) &&
      (maskf1 & 1))
    if ((coord_f0[1]*coord_f1[0] - coord_f0[0]*coord_f1[1]) < 0)
      return false;

  // edge 1: 0--2
  if ((maskf0 & 1) &&
      (maskf1 & 4))
    if ((coord_f0[2]*coord_f1[0] - coord_f0[0]*coord_f1[2]) > 0)
      return false;

  if ((maskf0 & 4) &&
      (maskf1 & 1))
    if ((coord_f0[2]*coord_f1[0] - coord_f0[0]*coord_f1[2]) < 0)
      return false;

  // edge 2: 0--3
  if ((maskf0 & 1) &&
      (maskf1 & 8))
    if ((coord_f0[3]*coord_f1[0] - coord_f0[0]*coord_f1[3]) > 0)
      return false;

  if ((maskf0 & 8) &&
      (maskf1 & 1))
    if ((coord_f0[3]*coord_f1[0] - coord_f0[0]*coord_f1[3]) < 0)
      return false;

  // edge 3: 1--2
  if ((maskf0 & 2) &&
      (maskf1 & 4))
    if ((coord_f0[2]*coord_f1[1] - coord_f0[1]*coord_f1[2]) > 0)
      return false;

  if ((maskf0 & 4) &&
      (maskf1 & 2))
    if ((coord_f0[2]*coord_f1[1] - coord_f0[1]*coord_f1[2]) < 0)
      return false;

  // edge 4: 1--3
  if ((maskf0 & 2) &&
      (maskf1 & 8))
    if ((coord_f0[3]*coord_f1[1] - coord_f0[1]*coord_f1[3]) > 0)
      return false;

  if ((maskf0 & 8) &&
      (maskf1 & 2))
    if ((coord_f0[3]*coord_f1[1] - coord_f0[1]*coord_f1[3]) < 0)
      return false;

  // edge 5: 2--3
  if ((maskf0 & 4) &&
      (maskf1 & 8))
    if ((coord_f0[3]*coord_f1[2] - coord_f0[2]*coord_f1[3]) > 0)
      return false;

  if ((maskf0 & 8) &&
      (maskf1 & 4))
    if ((coord_f0[3]*coord_f1[2] - coord_f0[2]*coord_f1[3]) < 0)
      return false;

  // Now there exists a separating plane supported by the edge shared
  // by f0 and f1.
  return true;
}
//-----------------------------------------------------------------------------
