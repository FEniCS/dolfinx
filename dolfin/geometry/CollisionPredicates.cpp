// Copyright (C) 2014-2017 Anders Logg, August Johansson and Benjamin Kehlet
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
// Last changed: 2017-10-09

#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/CellType.h>
#include "predicates.h"
#include "Point.h"
#include "CollisionPredicates.h"
#include "GeometryTools.h"

#include "CGALExactArithmetic.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
// High-level collision detection predicates
//-----------------------------------------------------------------------------
bool CollisionPredicates::collides(const MeshEntity& entity,
				   const Point& point)
{
  // Intersection is only implemented for simplex meshes
  if (!entity.mesh().type().is_simplex())
  {
    dolfin_error("Cell.cpp",
		 "intersect cell and point",
		 "Intersection is only implemented for simplex meshes");
  }

  // Get data
  const MeshGeometry& g = entity.mesh().geometry();
  const unsigned int* v = entity.entities(0);
  const std::size_t tdim = entity.mesh().topology().dim();
  const std::size_t gdim = entity.mesh().geometry().dim();

  // Pick correct specialized implementation
  if (tdim == 1 && gdim == 1)
    return collides_segment_point_1d(g.point(v[0])[0], g.point(v[1])[0], point[0]);

  if (tdim == 1 && gdim == 2)
    return collides_segment_point_2d(g.point(v[0]), g.point(v[1]), point);

  if (tdim == 1 && gdim == 3)
    return collides_segment_point_3d(g.point(v[0]), g.point(v[1]), point);

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
    return collides_tetrahedron_point_3d(g.point(v[0]),
                                         g.point(v[1]),
                                         g.point(v[2]),
                                         g.point(v[3]),
                                         point);

  dolfin_error("CollisionPredicates.cpp",
               "compute entity-point collision",
               "Not implemented for dimensions %d / %d", tdim, gdim);

  return false;
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::collides(const MeshEntity& entity_0,
				   const MeshEntity& entity_1)
{
  // Intersection is only implemented for simplex meshes
  if (!entity_0.mesh().type().is_simplex() ||
      !entity_1.mesh().type().is_simplex())
  {
    dolfin_error("Cell.cpp",
		 "intersect cell and point",
		 "intersection is only implemented for simplex meshes");
  }

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
  {
    return collides_segment_segment(g0.point(v0[0]),
                                    g0.point(v0[1]),
                                    g1.point(v1[0]),
                                    g1.point(v1[1]),
                                    gdim);
  }

  if (d0 == 1 && d1 == 2)
  {
    return collides_triangle_segment(g1.point(v1[0]),
				     g1.point(v1[1]),
				     g1.point(v1[2]),
				     g0.point(v0[0]),
				     g0.point(v0[1]),
				     gdim);
  }

  if (d0 == 2 && d1 == 1)
  {
    return collides_triangle_segment(g0.point(v0[0]),
                                     g0.point(v0[1]),
                                     g0.point(v0[2]),
                                     g1.point(v1[0]),
                                     g1.point(v1[1]),
                                     gdim);
  }

  if (d0 == 2 && d1 == 2)
  {
    return collides_triangle_triangle(g0.point(v0[0]),
                                      g0.point(v0[1]),
                                      g0.point(v0[2]),
                                      g1.point(v1[0]),
                                      g1.point(v1[1]),
                                      g1.point(v1[2]),
                                      gdim);
  }

  if (d0 == 2 && d1 == 3)
  {
    return collides_tetrahedron_triangle_3d(g1.point(v1[0]),
                                            g1.point(v1[1]),
                                            g1.point(v1[2]),
                                            g1.point(v1[3]),
                                            g0.point(v0[0]),
                                            g0.point(v0[1]),
                                            g0.point(v0[2]));
  }

  if (d0 == 3 && d1 == 2)
  {
    return collides_tetrahedron_triangle_3d(g0.point(v0[0]),
                                            g0.point(v0[1]),
                                            g0.point(v0[2]),
                                            g0.point(v0[3]),
                                            g1.point(v1[0]),
                                            g1.point(v1[1]),
                                            g1.point(v1[2]));
  }

  if (d0 == 3 && d1 == 3)
  {
    return collides_tetrahedron_tetrahedron_3d(g0.point(v0[0]),
                                               g0.point(v0[1]),
                                               g0.point(v0[2]),
                                               g0.point(v0[3]),
                                               g1.point(v1[0]),
                                               g1.point(v1[1]),
                                               g1.point(v1[2]),
                                               g1.point(v1[3]));
  }

  dolfin_error("CollisionPredicates.cpp",
               "compute entity-entity collision",
               "Not implemented for topological dimensions %d / %d and geometrical dimension %d", d0, d1, gdim);

  return false;
}
//-----------------------------------------------------------------------------
// Low-level collision detection predicates
//-----------------------------------------------------------------------------
bool CollisionPredicates::collides_segment_point(const Point& p0,
                                                 const Point& p1,
                                                 const Point& point,
                                                 std::size_t gdim)
{
  switch (gdim)
  {
  case 1:
    return collides_segment_point_1d(p0[0], p1[0], point[0]);
  case 2:
    return collides_segment_point_2d(p0, p1, point);
  case 3:
    return collides_segment_point_3d(p0, p1, point);
  default:
    dolfin_error("CollisionPredicates.cpp",
		 "call collides_segment_point",
		 "Unknown dimension (only implemented for dimension 2 and 3");
  }
  return false;
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::collides_segment_segment(const Point& p0,
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
    dolfin_error("CollisionPredicates.cpp",
		 "compute segment-segment collision ",
		 "Unknown dimension (Implemented for dimension 1, 2 and 3)");
  }
  return false;
}
//------------------------------------------------------------------------------
bool CollisionPredicates::collides_triangle_point(const Point& p0,
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
    dolfin_error("CollisionPredicates.cpp",
		 "compute triangle-point collision ",
		 "Implemented only for dimension 2 and 3.");
  }
  return false;
}
//------------------------------------------------------------------------------
bool CollisionPredicates::collides_triangle_segment(const Point& p0,
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
    dolfin_error("CollisionPredicates.cpp",
		 "compute triangle-segment collision ",
		 "Implmented only for dimension 2 and 3.");
  }
  return false;
}
//------------------------------------------------------------------------------
bool CollisionPredicates::collides_triangle_triangle(const Point& p0,
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
    dolfin_error("CollisionPredicates.cpp",
		 "compute triangle-triangle collision ",
		 "Implmented only for dimension 2 and 3.");
  }
  return false;
}



    //--- Low-level collision detection predicates ---
//------------------------------------------------------------------------------
bool CollisionPredicates::collides_segment_point_1d(double p0,
				      double p1,
				      double point)
{
  // FIXME: Skip CGAL for now
  return _collides_segment_point_1d(p0, p1, point);
}
//------------------------------------------------------------------------------
bool CollisionPredicates::collides_segment_point_2d(const Point& p0,
				      const Point& p1,
				      const Point& point)
{
  return CHECK_CGAL(_collides_segment_point_2d(p0, p1, point),
		    cgal_collides_segment_point_2d(p0, p1, point));
}
//------------------------------------------------------------------------------
bool CollisionPredicates::collides_segment_point_3d(const Point& p0,
						    const Point& p1,
						    const Point& point)
{
  return CHECK_CGAL(_collides_segment_point_3d(p0, p1, point),
		    cgal_collides_segment_point_3d(p0, p1, point));
}
//------------------------------------------------------------------------------
bool CollisionPredicates::collides_segment_segment_1d(double p0,
						      double p1,
						      double q0,
						      double q1)
{
  return _collides_segment_segment_1d(p0, p1, q0, q1);
}
//------------------------------------------------------------------------------
bool CollisionPredicates::collides_segment_segment_2d(const Point& p0,
						      const Point& p1,
						      const Point& q0,
						      const Point& q1)
{
  return CHECK_CGAL(_collides_segment_segment_2d(p0, p1, q0, q1),
		    cgal_collides_segment_segment_2d(p0, p1, q0, q1));
}
//------------------------------------------------------------------------------
bool CollisionPredicates::collides_segment_segment_3d(const Point& p0,
						      const Point& p1,
						      const Point& q0,
						      const Point& q1)
{
  return CHECK_CGAL(_collides_segment_segment_3d(p0, p1, q0, q1),
		    cgal_collides_segment_segment_3d(p0, p1, q0, q1));
}
//------------------------------------------------------------------------------
bool CollisionPredicates::collides_triangle_point_2d(const Point& p0,
						     const Point& p1,
						     const Point& p2,
						     const Point& point)
{
  return CHECK_CGAL(_collides_triangle_point_2d(p0, p1, p2, point),
		    cgal_collides_triangle_point_2d(p0, p1, p2, point));
}
//------------------------------------------------------------------------------
bool CollisionPredicates::collides_triangle_point_3d(const Point& p0,
						     const Point& p1,
						     const Point& p2,
						     const Point& point)
{
  return CHECK_CGAL(_collides_triangle_point_3d(p0, p1, p2, point),
		    cgal_collides_triangle_point_3d(p0, p1, p2, point));
}
//------------------------------------------------------------------------------
bool CollisionPredicates::collides_triangle_segment_2d(const Point& p0,
						       const Point& p1,
						       const Point& p2,
						       const Point& q0,
						       const Point& q1)
{
  return CHECK_CGAL(_collides_triangle_segment_2d(p0, p1, p2, q0, q1),
		    cgal_collides_triangle_segment_2d(p0, p1, p2, q0, q1));
}
//------------------------------------------------------------------------------
bool CollisionPredicates::collides_triangle_segment_3d(const Point& p0,
						       const Point& p1,
						       const Point& p2,
						       const Point& q0,
						       const Point& q1)
{
  return CHECK_CGAL(_collides_triangle_segment_3d(p0, p1, p2, q0, q1),
		    cgal_collides_triangle_segment_3d(p0, p1, p2, q0, q1));
}
//------------------------------------------------------------------------------
bool CollisionPredicates::collides_triangle_triangle_2d(const Point& p0,
							const Point& p1,
							const Point& p2,
							const Point& q0,
							const Point& q1,
							const Point& q2)
{
  return CHECK_CGAL(_collides_triangle_triangle_2d(p0, p1, p2, q0, q1, q2),
		    cgal_collides_triangle_triangle_2d(p0, p1, p2, q0, q1, q2));
}
//------------------------------------------------------------------------------
bool CollisionPredicates::collides_triangle_triangle_3d(const Point& p0,
							const Point& p1,
							const Point& p2,
							const Point& q0,
							const Point& q1,
							const Point& q2)
{
  return CHECK_CGAL(_collides_triangle_triangle_3d(p0, p1, p2, q0, q1, q2),
		    cgal_collides_triangle_triangle_3d(p0, p1, p2, q0, q1, q2));
}
//------------------------------------------------------------------------------
bool CollisionPredicates::collides_tetrahedron_point_3d(const Point& p0,
							const Point& p1,
							const Point& p2,
							const Point& p3,
							const Point& point)
{
  return CHECK_CGAL(_collides_tetrahedron_point_3d(p0, p1, p2, p3, point),
		    cgal_collides_tetrahedron_point_3d(p0, p1, p2, p3, point));
}
//------------------------------------------------------------------------------
bool CollisionPredicates::collides_tetrahedron_segment_3d(const Point& p0,
							  const Point& p1,
							  const Point& p2,
							  const Point& p3,
							  const Point& q0,
							  const Point& q1)
{
  return CHECK_CGAL(_collides_tetrahedron_segment_3d(p0, p1, p2, p3, q0, q1),
		    cgal_collides_tetrahedron_segment_3d(p0, p1, p2, p3, q0, q1));
}
//------------------------------------------------------------------------------
bool CollisionPredicates::collides_tetrahedron_triangle_3d(const Point& p0,
							   const Point& p1,
							   const Point& p2,
							   const Point& p3,
							   const Point& q0,
							   const Point& q1,
							   const Point& q2)
{
  return CHECK_CGAL(_collides_tetrahedron_triangle_3d(p0, p1, p2, p3, q0, q1, q2),
		    cgal_collides_tetrahedron_triangle_3d(p0, p1, p2, p3, q0, q1, q2));
}
//------------------------------------------------------------------------------
bool CollisionPredicates::collides_tetrahedron_tetrahedron_3d(const Point& p0,
							      const Point& p1,
							      const Point& p2,
							      const Point& p3,
							      const Point& q0,
							      const Point& q1,
							      const Point& q2,
							      const Point& q3)
{
  return CHECK_CGAL(_collides_tetrahedron_tetrahedron_3d(p0, p1, p2, p3, q0, q1, q2, q3),
		    cgal_collides_tetrahedron_tetrahedron_3d(p0, p1, p2, p3, q0, q1, q2, q3));
}
//-----------------------------------------------------------------------------
// Implementation of private members
//-----------------------------------------------------------------------------
bool CollisionPredicates::_collides_segment_point_1d(double p0,
                                                     double p1,
                                                     double point)
{
  if (p0 > p1)
    std::swap(p0, p1);
  return p0 <= point and point <= p1;
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::_collides_segment_point_2d(const Point& p0,
                                                     const Point& p1,
                                                     const Point& point)
{
  const double orientation = orient2d(p0, p1, point);

  const Point dp = p1 - p0;
  const double segment_length = dp.squared_norm();

  return orientation == 0.0 &&
    (point-p0).squared_norm() <= segment_length &&
    (point-p1).squared_norm() <= segment_length &&
    dp.dot(p1-point) >= 0.0 && dp.dot(point-p0) >= 0.0;
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::_collides_segment_point_3d(const Point& p0,
                                                     const Point& p1,
                                                     const Point& point)
{

  if (point == p0 or point == p1)
    return true;

  // Poject to reduce to three 2d problems
  const double det_xy = orient2d(p0, p1, point);

  if (det_xy == 0.0)
  {
    const std::array<std::array<double, 2>, 3> xz = {{ {{p0.x(), p0.z()}},
						       {{p1.x(), p1.z()}},
						       {{point.x(), point.z()}} }};
    const double det_xz = _orient2d(xz[0].data(),
				    xz[1].data(),
				    xz[2].data());

    if (det_xz == 0.0)
    {
      const std::array<std::array<double, 2>, 3> yz = {{ {{p0.y(), p0.z()}},
                                                         {{p1.y(), p1.z()}},
                                                         {{point.y(), point.z()}} }};
      const double det_yz = _orient2d(yz[0].data(),
				      yz[1].data(),
				      yz[2].data());

      if (det_yz == 0.0)
      {
        // Point is aligned with segment
        const double length = (p0 - p1).squared_norm();
        return (point-p0).squared_norm() <= length and
	  (point-p1).squared_norm() <= length;
      }
    }
  }

  return false;
}
//------------------------------------------------------------------------------
bool CollisionPredicates::_collides_segment_segment_1d(double p0,
                                                       double p1,
                                                       double q0,
                                                       double q1)
{
  // Get range
  const double a0 = std::min(p0, p1);
  const double b0 = std::max(p0, p1);
  const double a1 = std::min(q0, q1);
  const double b1 = std::max(q0, q1);

  // Check for collision
  const double dx = std::min(b0 - a0, b1 - a1);
  return b1 >= a0 - dx && a1 <= b0 + dx;
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::_collides_segment_segment_2d(const Point& p0,
                                                       const Point& p1,
                                                       const Point& q0,
                                                       const Point& q1)
{
  // FIXME: Optimize by avoiding redundant calls to orient2d

  if (collides_segment_point_2d(p0, p1, q0))
    return true;
  if (collides_segment_point_2d(p0, p1, q1))
    return true;
  if (collides_segment_point_2d(q0, q1, p0))
    return true;
  if (collides_segment_point_2d(q0, q1, p1))
    return true;

  // Points must be on different sides
  if (((orient2d(q0, q1, p0) > 0.0) xor (orient2d(q0, q1, p1) > 0.0)) and
      ((orient2d(p0, p1, q0) > 0.0) xor (orient2d(p0, p1, q1) > 0.0)))
    return true;
  else
    return false;
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::_collides_segment_segment_3d(const Point& p0,
                                                       const Point& p1,
                                                       const Point& q0,
                                                       const Point& q1)
{
  // Vertex collisions
  if (p0 == q0 || p0 == q1 || p1 == q0 || p1 == q1)
    return true;

  if (collides_segment_point_3d(p0, p1, q0) or
      collides_segment_point_3d(p0, p1, q1) or
      collides_segment_point_3d(q0, q1, p0) or
      collides_segment_point_3d(q0, q1, p1))
  {
    return true;
  }

  // Determinant must be zero
  const double det = orient3d(p0, p1, q0, q1);

  if (det < 0.0 or det > 0.0)
    return false;

  // Now we know that the segments are in the same plane. This means
  // that they can be parallel, or even collinear.

  // Check for collinearity
  const Point u = GeometryTools::cross_product(p0, p1, q0);
  if (u[0] == 0.0 and u[1] == 0.0 and u[2] == 0.0)
  {
    const Point v = GeometryTools::cross_product(p0, p1, q1);
    if (v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0)
    {
      // Now we know that the segments are collinear
      if ((p0-q0).squared_norm() <= (q1-q0).squared_norm() and
	  (p0-q1).squared_norm() <= (q0-q1).squared_norm())
	return true;

      if ((p1-q0).squared_norm() <= (q1-q0).squared_norm() and
	  (p1-q1).squared_norm() <= (q0-q1).squared_norm())
	return true;

      if ((q0-p0).squared_norm() <= (p1-p0).squared_norm() and
	  (q0-p1).squared_norm() <= (p0-p1).squared_norm())
	return true;

      if ((q1-p0).squared_norm() <= (p1-p0).squared_norm() and
	  (q1-p1).squared_norm() <= (p0-p1).squared_norm())
	return true;
    }
  }

  // Segments are not collinear, but in the same plane
  // Try to reduce to 2d by elimination

  for (std::size_t d = 0; d < 3; ++d)
  {
    if (p0[d] == p1[d] and p0[d] == q0[d] and p0[d] == q1[d])
    {
      const std::array<std::array<std::size_t, 2>, 3> dims = {{ {{1, 2}},
                                                                {{0, 2}},
                                                                {{0, 1}} }};
      const Point p0_2d(p0[dims[d][0]], p0[dims[d][1]]);
      const Point p1_2d(p1[dims[d][0]], p1[dims[d][1]]);
      const Point q0_2d(q0[dims[d][0]], q0[dims[d][1]]);
      const Point q1_2d(q1[dims[d][0]], q1[dims[d][1]]);

      return collides_segment_segment_2d(p0_2d, p1_2d, q0_2d, q1_2d);
    }
  }

  return false;
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::_collides_triangle_point_2d(const Point& p0,
                                                      const Point& p1,
                                                      const Point& p2,
                                                      const Point& point)
{
  const double ref = orient2d(p0, p1, p2);

  if (ref > 0.0)
  {
    return (orient2d(p1, p2, point) >= 0.0 and
	    orient2d(p2, p0, point) >= 0.0 and
	    orient2d(p0, p1, point) >= 0.0);
  }
  else if (ref < 0.0)
  {
    return (orient2d(p1, p2, point) <= 0.0 and
	    orient2d(p2, p0, point) <= 0.0 and
	    orient2d(p0, p1, point) <= 0.0);
  }
  else
  {
    return ((orient2d(p0, p1, point) == 0.0 and
	     collides_segment_point_1d(p0[0], p1[0], point[0]) and
	     collides_segment_point_1d(p0[1], p1[1], point[1])) or
	    (orient2d(p1, p2, point) == 0.0 and
	     collides_segment_point_1d(p1[0], p2[0], point[0]) and
	     collides_segment_point_1d(p1[1], p2[1], point[1])) or
	    (orient2d(p2, p0, point) == 0.0 and
	     collides_segment_point_1d(p2[0], p0[0], point[0]) and
	     collides_segment_point_1d(p2[1], p0[1], point[1])));
  }
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::_collides_triangle_point_3d(const Point& p0,
                                                      const Point& p1,
                                                      const Point& p2,
                                                      const Point& point)
{
  if (p0 == point or p1 == point or p2 == point)
    return true;

  const double tet_det = orient3d(p0, p1, p2, point);

  if (tet_det < 0.0 or tet_det > 0.0)
    return false;

  // Use normal
  const Point n = GeometryTools::cross_product(p0, p1, p2);

  return !(n.dot(GeometryTools::cross_product(point, p0, p1)) < 0.0 or
	   n.dot(GeometryTools::cross_product(point, p2, p0)) < 0.0 or
	   n.dot(GeometryTools::cross_product(point, p1, p2)) < 0.0);
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::_collides_triangle_segment_2d(const Point& p0,
                                                        const Point& p1,
                                                        const Point& p2,
                                                        const Point& q0,
                                                        const Point& q1)
{
  // FIXME: Optimize by avoiding redundant calls to orient2d

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
bool CollisionPredicates::_collides_triangle_segment_3d(const Point& r,
                                                        const Point& s,
                                                        const Point& t,
                                                        const Point& a,
                                                        const Point& b)
{
  // FIXME: Optimize by avoiding redundant calls to orient3d

  // Compute correspondic tetrahedra determinants
  const double rsta = orient3d(r, s, t, a);
  const double rstb = orient3d(r, s, t, b);

  // Check if a and b are on same side of triangle rst
  if ((rsta < 0.0 and rstb < 0.0) or
      (rsta > 0.0 and rstb > 0.0))
    return false;

  // We check triangle point first. We use this below.
  if (collides_triangle_point_3d(r, s, t, a))
    return true;

  if (collides_triangle_point_3d(r, s, t, b))
    return true;

  // Now we know a and b are either on different sides or in the same
  // plane (in which case rsta = rstb = 0). Check if intersection is
  // in triangle by creating some other tets.

  if (rsta == 0.0 and rstb == 0.0)
  {
    // Since we have checked that the points does not collide, the
    // segment is either completely outside the triangle, or we have a
    // collision over edges.

    // FIXME: To avoid collision over edges, maybe we can test if both
    // a and b are on the same side of one of the edges rs, rt or st.

    if (collides_segment_segment_3d(r, s, a, b))
      return true;
    if (collides_segment_segment_3d(r, t, a, b))
      return true;
    if (collides_segment_segment_3d(s, t, a, b))
      return true;

    return false;
  }
  else
  {
    // Temporarily flip a and b to make sure a is above
    Point _a = a;
    Point _b = b;
    if (rsta < 0.0)
      std::swap(_a, _b);

    const double rasb = orient3d(r, _a, s, _b);
    if (rasb < 0)
      return false;

    const double satb = orient3d(s, _a, t, _b);
    if (satb < 0)
      return false;

    const double tarb = orient3d(t, _a, r, _b);
    if (tarb < 0)
      return false;
  }

  return true;
}
//------------------------------------------------------------------------------
bool CollisionPredicates::_collides_triangle_triangle_2d(const Point& p0,
                                                         const Point& p1,
                                                         const Point& p2,
                                                         const Point& q0,
                                                         const Point& q1,
                                                         const Point& q2)
{
  // FIXME: Optimize by avoiding redundant calls to orient2d

  // Pack points as vectors
  const std::array<Point, 3> tri_0 = {{p0, p1, p2}};
  const std::array<Point, 3> tri_1 = {{q0, q1, q2}};

  const bool s0 = std::signbit(orient2d(p0, p1, p2));
  const bool s1 = std::signbit(orient2d(q0, q1, q2));

  for (std::size_t i = 0; i < 3; ++i)
  {
    if ((s0 and
    	 orient2d(tri_0[0], tri_0[1], tri_1[i]) <= 0.0 and
    	 orient2d(tri_0[1], tri_0[2], tri_1[i]) <= 0.0 and
    	 orient2d(tri_0[2], tri_0[0], tri_1[i]) <= 0.0)
	or
    	(!s0 and
    	 orient2d(tri_0[0], tri_0[1], tri_1[i]) >= 0.0 and
    	 orient2d(tri_0[1], tri_0[2], tri_1[i]) >= 0.0 and
    	 orient2d(tri_0[2], tri_0[0], tri_1[i]) >= 0.0))
    {
      return true;
    }

    if ((s1 and
    	 orient2d(tri_1[0], tri_1[1], tri_0[i]) <= 0.0 and
    	 orient2d(tri_1[1], tri_1[2], tri_0[i]) <= 0.0 and
    	 orient2d(tri_1[2], tri_1[0], tri_0[i]) <= 0.0)
	or
    	(!s1 and
    	 orient2d(tri_1[0], tri_1[1], tri_0[i]) >= 0.0 and
    	 orient2d(tri_1[1], tri_1[2], tri_0[i]) >= 0.0 and
    	 orient2d(tri_1[2], tri_1[0], tri_0[i]) >= 0.0))
    {
      return true;
    }
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
      if (collides_segment_segment_2d(p0, q0, p1, q1))
        return true;
    }
  }

  return false;
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::_collides_triangle_triangle_3d(const Point& p0,
                                                         const Point& p1,
                                                         const Point& p2,
                                                         const Point& q0,
                                                         const Point& q1,
                                                         const Point& q2)
{
  // FIXME: Optimize by avoiding redundant calls to orient3d

  // Pack points as vectors
  const std::array<Point, 3> tri_0 = {{p0, p1, p2}};
  const std::array<Point, 3> tri_1 = {{q0, q1, q2}};

  // First test edge-face collisions
  for (std::size_t i = 0; i < 3; ++i)
  {
    const std::size_t j = (i + 1) % 3;

    if (collides_triangle_segment_3d(p0, p1, p2, tri_1[i], tri_1[j]))
      return true;

    if (collides_triangle_segment_3d(q0, q1, q2, tri_0[i], tri_0[j]))
      return true;
  }

  // Test edge-edge collisions
  for (std::size_t i0 = 0; i0 < 3; i0++)
  {
    const std::size_t j0 = (i0 + 1) % 3;
    for (std::size_t i1 = 0; i1 < 3; i1++)
    {
      const std::size_t j1 = (i1 + 1) % 3;
      if (collides_segment_segment_3d(tri_0[i0], tri_0[j0],
				      tri_1[i1], tri_1[j1]))
      {
	return true;
      }
    }
  }

  // FIXME
  // Test point-face collisions (could also be detected by
  // triangle_segment collision above)
  for (std::size_t i = 0; i < 3; ++i)
  {
    if (collides_triangle_point_3d(p0, p1, p2, tri_1[i]))
      return true;

    if (collides_triangle_point_3d(q0, q1, q2, tri_0[i]))
      return true;
  }

  return false;
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::_collides_tetrahedron_point_3d(const Point& p0,
                                                         const Point& p1,
                                                         const Point& p2,
                                                         const Point& p3,
                                                         const Point& point)
{
  const double ref = orient3d(p0, p1, p2, p3);

  if (ref > 0.0)
  {
    return (orient3d(p0, p1, p2, point) >= 0.0 and
	    orient3d(p0, p3, p1, point) >= 0.0 and
	    orient3d(p0, p2, p3, point) >= 0.0 and
	    orient3d(p1, p3, p2, point) >= 0.0);
  }
  else if (ref < 0.0)
  {
    return (orient3d(p0, p1, p2, point) <= 0.0 and
	    orient3d(p0, p3, p1, point) <= 0.0 and
	    orient3d(p0, p2, p3, point) <= 0.0 and
	    orient3d(p1, p3, p2, point) <= 0.0);
  }
  else
  {
    dolfin_error("CollisionPredicates.cpp",
		 "compute tetrahedron point collision",
		 "Not implemented for degenerate tetrahedron");
  }

  return false;
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::_collides_tetrahedron_segment_3d(const Point& p0,
                                                           const Point& p1,
                                                           const Point& p2,
                                                           const Point& p3,
                                                           const Point& q0,
                                                           const Point& q1)
{
  // FIXME: Optimize by avoiding redundant calls to orient3d

  // Segment vertex in tetrahedron collision
  if (collides_tetrahedron_point_3d(p0, p1, p2, p3, q0))
    return true;
  if (collides_tetrahedron_point_3d(p0, p1, p2, p3, q1))
    return true;

  // Triangle-segment collision tests
  if (collides_triangle_segment_3d(p1, p2, p3, q0, q1))
    return true;
  if (collides_triangle_segment_3d(p0, p2, p3, q0, q1))
    return true;
  if (collides_triangle_segment_3d(p0, p1, p3, q0, q1))
    return true;
  if (collides_triangle_segment_3d(p0, p1, p2, q0, q1))
    return true;

  return false;
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::_collides_tetrahedron_triangle_3d(const Point& p0,
                                                            const Point& p1,
                                                            const Point& p2,
                                                            const Point& p3,
                                                            const Point& q0,
                                                            const Point& q1,
                                                            const Point& q2)
{
  // FIXME: Optimize by avoiding redundant calls to orient3d

  // Triangle vertex in tetrahedron collision
  if (collides_tetrahedron_point_3d(p0, p1, p2, p3, q0))
    return true;
  if (collides_tetrahedron_point_3d(p0, p1, p2, p3, q1))
    return true;
  if (collides_tetrahedron_point_3d(p0, p1, p2, p3, q2))
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
bool CollisionPredicates::_collides_tetrahedron_tetrahedron_3d(const Point& p0,
                                                               const Point& p1,
                                                               const Point& p2,
                                                               const Point& p3,
                                                               const Point& q0,
                                                               const Point& q1,
                                                               const Point& q2,
                                                               const Point& q3)
{
  // FIXME: Optimize by avoiding redundant calls to orient3d

  const std::array<Point, 4> tetp = {{p0, p1, p2, p3}};
  const std::array<Point, 4> tetq = {{q0, q1, q2, q3}};

  // Triangle face collisions
  const std::array<std::array<std::size_t, 3>, 4> faces = {{ {{1, 2, 3}},
                                                             {{0, 2, 3}},
                                                             {{0, 1, 3}},
                                                             {{0, 1, 2}} }};
  for (std::size_t i = 0; i < 4; ++i)
  {
    for (std::size_t j = 0; j < 4; ++j)
    {
      if (collides_triangle_triangle_3d(tetp[faces[i][0]], tetp[faces[i][1]], tetp[faces[i][2]],
					tetq[faces[j][0]], tetq[faces[j][1]], tetq[faces[j][2]]))
      {
	return true;
      }
    }
  }

  // Vertex in tetrahedron collision
  if (collides_tetrahedron_point_3d(p0, p1, p2, p3, q0))
    return true;
  if (collides_tetrahedron_point_3d(p0, p1, p2, p3, q1))
    return true;
  if (collides_tetrahedron_point_3d(p0, p1, p2, p3, q2))
    return true;
  if (collides_tetrahedron_point_3d(p0, p1, p2, p3, q3))
    return true;

  if (collides_tetrahedron_point_3d(q0, q1, q2, q3, p0))
    return true;
  if (collides_tetrahedron_point_3d(q0, q1, q2, q3, p1))
    return true;
  if (collides_tetrahedron_point_3d(q0, q1, q2, q3, p2))
    return true;
  if (collides_tetrahedron_point_3d(q0, q1, q2, q3, p3))
    return true;

  return false;
}
//-----------------------------------------------------------------------------
