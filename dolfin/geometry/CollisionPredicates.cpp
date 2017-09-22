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
// Last changed: 2017-09-22

#include <dolfin/mesh/MeshEntity.h>
#include "predicates.h"
#include "Point.h"
#include "CollisionPredicates.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
// High-level collision detection predicates
//-----------------------------------------------------------------------------
bool CollisionPredicates::collides(const MeshEntity& entity,
           const Point& point)
{
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
    return collides_segment_segment(g0.point(v0[0]),
                                    g0.point(v0[1]),
                                    g1.point(v1[0]),
                                    g1.point(v1[1]),
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
    return collides_tetrahedron_triangle_3d(g1.point(v1[0]),
                                            g1.point(v1[1]),
                                            g1.point(v1[2]),
                                            g1.point(v1[3]),
                                            g0.point(v0[0]),
                                            g0.point(v0[1]),
                                            g0.point(v0[2]));

  if (d0 == 3 && d1 == 2)
    return collides_tetrahedron_triangle_3d(g0.point(v0[0]),
                                            g0.point(v0[1]),
                                            g0.point(v0[2]),
                                            g0.point(v0[3]),
                                            g1.point(v1[0]),
                                            g1.point(v1[1]),
                                            g1.point(v1[2]));

  if (d0 == 3 && d1 == 3)
    return collides_tetrahedron_tetrahedron_3d(g0.point(v0[0]),
                                               g0.point(v0[1]),
                                               g0.point(v0[2]),
                                               g0.point(v0[3]),
                                               g1.point(v1[0]),
                                               g1.point(v1[1]),
                                               g1.point(v1[2]),
                                               g1.point(v1[3]));

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
     "collides_segment_point",
     "Unknown dimension (only implemented for dimension 2 and 3");
  }
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
    std::array<std::array<double, 2>, 3> xz = {{ { p0.x(), p0.z() },
               { p1.x(), p1.z() },
               { point.x(), point.z() } }};
    const double det_xz = _orient2d(xz[0].data(),
            xz[1].data(),
            xz[2].data());

    if (det_xz == 0.0)
    {
      std::array<std::array<double, 2>, 3> yz = {{ { p0.y(), p0.z() },
                 { p1.y(), p1.z() },
                 { point.y(), point.z() } }};
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
//-----------------------------------------------------------------------------
bool CollisionPredicates::_collides_interior_point_segment_2d(const Point& q0,
                                                              const Point& q1,
                                                              const Point& p)
{
  const double q0_q1_p = orient2d(q0, q1, p);
  const Point dq = q1-q0;
  const double segment_length = dq.squared_norm();

  return q0_q1_p == 0.0 && (p-q0).squared_norm() <= segment_length && (p-q1).squared_norm() <= segment_length && dq.dot(q1-p) > 0 && dq.dot(p-q0) > 0.0;
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::_collides_interior_point_segment_3d(const Point& q0,
                                                              const Point& q1,
                                                              const Point& p)
{
  // FIXME
  dolfin_error("CollisionPredicates",
         "_collides_interior_point_segment_3d",
         "Not implemented");
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

  // FIXME
  // // Check for collisions
  // const double dx = std::min(b0 - a0, b1 - a1);
  // const double eps = std::max(DOLFIN_EPS_LARGE, DOLFIN_EPS_LARGE*dx);
  // const bool result = b1 > a0 - eps && a1 < b0 + eps;

  const double dx = std::min(b0 - a0, b1 - a1);
  const bool result = b1 >= a0 - dx && a1 <= b0 + dx;

  return result;
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::_collides_segment_segment_2d(const Point& p0,
                                                       const Point& p1,
                                                       const Point& q0,
                                                       const Point& q1)
{
  if (collides_segment_point_2d(p0, p1, q0)) return true;
  if (collides_segment_point_2d(p0, p1, q1)) return true;
  if (collides_segment_point_2d(q0, q1, p0)) return true;
  if (collides_segment_point_2d(q0, q1, p1)) return true;

  const double q0_q1_p0 = orient2d(q0, q1, p0);
  const double q0_q1_p1 = orient2d(q0, q1, p1);
  const double p0_p1_q0 = orient2d(p0, p1, q0);
  const double p0_p1_q1 = orient2d(p0, p1, q1);

  // Products must be strictly smaller
  return q0_q1_p0*q0_q1_p1 < 0.0 && p0_p1_q0*p0_p1_q1 < 0.0;
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
    return true;

  // Determinant must be zero
  const double det = orient3d(p0, p1, q0, q1);

  if (det < 0. or det > 0.)
    return false;

  // Now we know that the segments are in the same plane. This means
  // that they can be parallel, or even collinear.

  // Check for collinearity
  const Point u = cross_product(p0, p1, q0);
  if (u[0] == 0. and u[1] == 0. and u[2] == 0.)
  {
    const Point v = cross_product(p0, p1, q1);
    if (v[0] == 0. and v[1] == 0. and v[2] == 0.)
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
      const std::array<std::array<std::size_t, 2>, 3> dims = {{ {1, 2}, {0, 2}, {0, 1} }};
      Point p0_2d(p0[dims[d][0]], p0[dims[d][1]]);
      Point p1_2d(p1[dims[d][0]], p1[dims[d][1]]);
      Point q0_2d(q0[dims[d][0]], q0[dims[d][1]]);
      Point q1_2d(q1[dims[d][0]], q1[dims[d][1]]);

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

  if (ref != 0.0)
  {
    if (ref*orient2d(p0, p1, point) >= 0.0 and
	ref*orient2d(p1, p2, point) >= 0.0 and
	ref*orient2d(p2, p0, point) >= 0.0)
      return true;
    else
      return false;
  }
  else
  {
    if ((orient2d(p0, p1, point) == 0.0 and
	 collides_segment_point_1d(p0[0], p1[0], point[0]) and
	 collides_segment_point_1d(p0[1], p1[1], point[1])) or
	(orient2d(p1, p2, point) == 0.0 and
	 collides_segment_point_1d(p1[0], p2[0], point[0]) and
	 collides_segment_point_1d(p1[1], p2[1], point[1])) or
	(orient2d(p2, p0, point) == 0.0 and
	 collides_segment_point_1d(p2[0], p0[0], point[0]) and
	 collides_segment_point_1d(p2[1], p0[1], point[1])))
      return true;
    else
      return false;
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

  // FIXME: The determinant should be exactly zero for the point to be
  // in the plane. However, if we take a triangle with vertices
  // (0,0,1), (1,1,1), (0,1,0) and check the point (1./3,2./3,2./3)
  // this gives a determinant of ~5.55112e-17
  //if (std::abs(det) > DOLFIN_EPS)
  if (tet_det < 0. or tet_det > 0.)
    return false;

  // // Check that the point is inside the triangle using barycentric coords
  // const double tri_det = cross_product_norm(p0, p1, p2);
  // std::cout<<std::setprecision(16)<<tri_det << ' '<<((p0-p2).cross(p1-p2)).norm()<<'\n';

  // // const double r = cross_product_norm(p0, p1, point);
  // // std::cout<<std::setprecision(16) << r << "  (r<0) " << (r<0) << " (r>tri_det) " << (r>tri_det)<< '\n';
  // // if (r < 0 or r > tri_det)
  // //   return false;

  // // const double s = cross_product_norm(p1, p2, point);
  // // std::cout<<std::setprecision(16) << s << "  (s<0) " << (s<0) << " (s>tri_det) " << (s>tri_det)<<'\n';
  // // std::cout<<std::setprecision(16) << (r+s) << "  (r+s<0) " << (r+s<0) << " (r+s>tri_det) " << (r+s>tri_det)<< '\n';
  // // if (s < 0 or s > tri_det or
  // //     (r+s) < 0 or (r+s) > tri_det)
  // //   return false;

  // const double r = cross_product_norm(p0, p1, point);
  // // std::cout << tools::plot3(p0)<<'\n'
  // //       << tools::plot3(p1)<<'\n'
  // //       << tools::plot3(point)<<'\n';
  // std::cout<<std::setprecision(16) << r << " (r>tri_det) " << (r>tri_det)<< '\n';
  // if (r > tri_det)
  //   return false;

  // const double s = cross_product_norm(p1, p2, point);
  // // std::cout << tools::plot3(p1)<<'\n'
  // //       << tools::plot3(p2)<<'\n'
  // //       << tools::plot3(point)<<'\n';
  // std::cout<<std::setprecision(16) << s << " (s>tri_det) " << (s>tri_det)<<'\n';
  // if (s > tri_det)
  //   return false;

  // const double t = cross_product_norm(p0, p2, point);
  // std::cout<<std::setprecision(16) << t << " (t>tri_det) " << (t>tri_det)<<'\n';
  // std::cout<<std::setprecision(16) << "r+s+t " << r+s+t << '\n';
  // if (t > tri_det or r+s+t>tri_det)
  //   return false;

  // return true;

  // const double tri_det_2 = cross_product(p0, p1, p2).squared_norm();

  // std::cout<<std::setprecision(16)<<std::scientific << "permute " << (tri_det_2==cross_product(p0, p2, p1).squared_norm()) <<  ' '<< (tri_det_2==cross_product(p2, p1, p0).squared_norm())<<'\n';


  // if (cross_product(p0, p1, point).squared_norm() > tri_det_2)
  //   return false;

  // if (cross_product(p1, p2, point).squared_norm() > tri_det_2)
  //   return false;

  // if (cross_product(p0, p2, point).squared_norm() > tri_det_2)
  //   return false;

  // std::cout<<"sqrt " << std::setprecision(16)<<std::scientific<< cross_product(p0, p1, point).norm()<<' '<< cross_product(p1, p2, point).norm()<<' '<< cross_product(p0, p2, point).norm()<<' '<< std::sqrt(tri_det_2)<<'\n';
  // std::cout << "sqrt sum < " << std::setprecision(16)<<std::scientific<< cross_product(p0, p1, point).norm()  + cross_product(p1, p2, point).norm()+ cross_product(p0, p2, point).norm()<<' '<< std::sqrt(tri_det_2)<<'\n';
  // std::cout<<std::setprecision(16)<<std::scientific<< cross_product(p0, p1, point).squared_norm()<<' '<< cross_product(p1, p2, point).squared_norm()<<' '<< cross_product(p0, p2, point).squared_norm()<<' '<< tri_det_2<<'\n';

  // if (cross_product(p0, p1, point).norm()
  //     + cross_product(p1, p2, point).norm()
  //     + cross_product(p0, p2, point).norm() > std::sqrt(tri_det_2))
  //   return false;

  // return true;


  // use normal
  const Point n = cross_product(p0, p1, p2);

  if (n.dot(cross_product(point, p0, p1)) < 0.0 or
      n.dot(cross_product(point, p2, p0)) < 0.0 or
      n.dot(cross_product(point, p1, p2)) < 0.0)
    return false;
  return true;

  // // FIXME
  // // Test: Reduce to 2d problem by taking the projection of the
  // // triangle onto the 2d plane xy, xz or yz that has the largest
  // // determinant
  // const double det_xy = std::abs(orient2d(p0,
  //            p1,
  //            p2));
  // std::cout << __FUNCTION__ << "  detxy " << det_xy <<std::endl;

  // std::array<Point, 3> xz = { Point(p0.x(), p0.z()),
  //            Point(p1.x(), p1.z()),
  //            Point(p2.x(), p2.z()) };
  // const double det_xz = std::abs(orient2d(xz[0],
  //            xz[1],
  //            xz[2]));
  // std::cout << __FUNCTION__ << "  detxz " << det_xz <<std::endl;

  // std::array<Point, 3> yz = { Point(p0.y(), p0.z()),
  //            Point(p1.y(), p1.z()),
  //            Point(p2.y(), p2.z()) };
  // const double det_yz = std::abs(orient2d(yz[0],
  //            yz[1],
  //            yz[2]));
  // std::cout << __FUNCTION__ << "  detyz " << det_yz <<std::endl;

  // // Check for degeneracy
  // dolfin_assert(det_xy > DOLFIN_EPS or
  //    det_xz > DOLFIN_EPS or
  //    det_yz > DOLFIN_EPS);

  // std::array<Point, 3> tri;
  // Point a;

  // if (det_xy > det_xz and det_xy > det_yz)
  // {
  //   tri = std::array<Point, 3>{ p0, p1, p2 };
  //   a[0] = point[0];
  //   a[1] = point[1];
  // }
  // else if (det_xz > det_xy and det_xz > det_yz)
  // {
  //   tri = xz;
  //   a[0] = point[0];
  //   a[1] = point[2];
  // }
  // else
  // {
  //   tri = yz;
  //   a[0] = point[1];
  //   a[1] = point[2];
  // }

  // // std::cout << tri[0]<<' '<<tri[1]<<' '<<tri[2]<<"    " << ' ' << a << std::endl;

  // // const bool collides_2d = collides_triangle_point_2d(tri[0], tri[1], tri[2], a);
  // // std::cout << "2d collision " << col2d << std::endl;

  // return collides_triangle_point_2d(tri[0], tri[1], tri[2], a);
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::_collides_triangle_segment_2d(const Point& p0,
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
bool CollisionPredicates::_collides_triangle_segment_3d(const Point& r,
                                                        const Point& s,
                                                        const Point& t,
                                                        const Point& a,
                                                        const Point& b)
{
  // Compute correspondic tetrahedra determinants
  const double rsta = orient3d(r, s, t, a);

  const double rstb = orient3d(r, s, t, b);

  // Check if a and b are on same side of triangle rst
  if ((rsta < 0.0 and rstb < 0.0) or
      (rsta > 0.0 and rstb > 0.0))
    return false;

  // We check triangle point first. We use this below.
  if (collides_triangle_point_3d(r, s, t, a))
  {
    return true;
  }

  if (collides_triangle_point_3d(r, s, t, b))
  {
    return true;
  }

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
    if (rsta < 0)
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
  // Pack points as vectors
  std::array<Point, 3> tri_0({p0, p1, p2});
  std::array<Point, 3> tri_1({q0, q1, q2});

  // Extract coordinates
  double t0[3][2] = {{p0[0], p0[1]}, {p1[0], p1[1]}, {p2[0], p2[1]}};
  double t1[3][2] = {{q0[0], q0[1]}, {q1[0], q1[1]}, {q2[0], q2[1]}};

  // Find all vertex-cell collisions
  const int s0 = std::signbit(_orient2d(t0[0], t0[1], t0[2])) == true ? -1 : 1;
  const int s1 = std::signbit(_orient2d(t1[0], t1[1], t1[2])) == true ? -1 : 1;

  for (std::size_t i = 0; i < 3; ++i)
  {
    if (s1*_orient2d(t1[0], t1[1], t0[i]) >= 0. and
    s1*_orient2d(t1[1], t1[2], t0[i]) >= 0. and
    s1*_orient2d(t1[2], t1[0], t0[i]) >= 0.)
      return true;

    if (s0*_orient2d(t0[0], t0[1], t1[i]) >= 0. and
    s0*_orient2d(t0[1], t0[2], t1[i]) >= 0. and
    s0*_orient2d(t0[2], t0[0], t1[i]) >= 0.)
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
  // Pack points as vectors
  std::array<Point, 3> tri_0({p0, p1, p2});
  std::array<Point, 3> tri_1({q0, q1, q2});

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
  return true;
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
  const double ref = orient3d(p0,
            p1,
            p2,
            p3);

  if (ref*orient3d(p0,
       p1,
       p2,
       point) >= 0.0 and
      ref*orient3d(p0,
       p3,
       p1,
       point) >= 0.0 and
      ref*orient3d(p0,
       p2,
       p3,
       point) >= 0.0 and
      ref*orient3d(p1,
       p3,
       p2,
       point) >= 0.0)
    return true;
  else
    return false;
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::_collides_interior_tetrahedron_point_3d(const Point& p0,
								  const Point& p1,
								  const Point& p2,
								  const Point& p3,
								  const Point& point)
{
  const double ref = orient3d(p0,
            p1,
            p2,
            p3);

  if (ref*orient3d(p0,
       p1,
       p2,
       point) > 0.0 and
      ref*orient3d(p0,
       p3,
       p1,
       point) > 0.0 and
      ref*orient3d(p0,
       p2,
       p3,
       point) > 0.0 and
      ref*orient3d(p1,
       p3,
       p2,
       point) > 0.0)
    return true;
  else
    return false;


  // // Check tetrahedron orientation
  // const int sign = std::signbit(orient3d(p0,
  //           p1,
  //           p2,
  //           p3));

  // // The point is inside if all tetrahedra formed have the same orientation
  // if (sign == std::signbit(orient3d(p0,
  //            p1,
  //            p2,
  //            point)) and
  //     sign == std::signbit(orient3d(p0,
  //            p3,
  //            p1,
  //            point)) and
  //     sign == std::signbit(orient3d(p0,
  //            p2,
  //            p3,
  //            point)) and
  //     sign == std::signbit(orient3d(p1,
  //            p3,
  //            p2,
  //            point)))
  //   return true;
  // else
  //   return false;
}
//-----------------------------------------------------------------------------
bool CollisionPredicates::_collides_tetrahedron_segment_3d(const Point& p0,
                                                           const Point& p1,
                                                           const Point& p2,
                                                           const Point& p3,
                                                           const Point& q0,
                                                           const Point& q1)
                                          {
  // std::cout << __FUNCTION__<<std::endl;

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
  const std::array<Point, 4> tetp = {{p0, p1, p2, p3}};
  const std::array<Point, 4> tetq = {{q0, q1, q2, q3}};

  // Triangle face collisions
  const std::array<std::array<std::size_t, 3>, 4> faces = {{ {1, 2, 3},
                   {0, 2, 3},
                   {0, 1, 3},
                   {0, 1, 2} }};
  for (std::size_t i = 0; i < 4; ++i)
    for (std::size_t j = 0; j < 4; ++j)
      if (collides_triangle_triangle_3d(tetp[faces[i][0]], tetp[faces[i][1]], tetp[faces[i][2]],
          tetq[faces[j][0]], tetq[faces[j][1]], tetq[faces[j][2]]))
  return true;

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
Point CollisionPredicates::cross_product(const Point& a,
                                         const Point& b,
                                         const Point& c)
{
  // Accurate cross product p = (a-c) x (b-c). See Shewchuk Lecture
  // Notes on Geometric Robustness.
  double ayz[2] = {a.y(), a.z()};
  double byz[2] = {b.y(), b.z()};
  double cyz[2] = {c.y(), c.z()};
  double azx[2] = {a.z(), a.x()};
  double bzx[2] = {b.z(), b.x()};
  double czx[2] = {c.z(), c.x()};
  double axy[2] = {a.x(), a.y()};
  double bxy[2] = {b.x(), b.y()};
  double cxy[2] = {c.x(), c.y()};
  Point p(_orient2d(ayz, byz, cyz),
	  _orient2d(azx, bzx, czx),
	  _orient2d(axy, bxy, cxy));
  return p;
}
//-----------------------------------------------------------------------------
double CollisionPredicates::cross_product_norm(const Point& a,
                                               const Point& b,
                                               const Point& c)
{
  // Accurate norm of cross product p = (a-c) x (b-c). See Shewchuk
  // Lecture Notes on Geometric Robustness.
  double ayz[2] = {a.y(), a.z()};
  double byz[2] = {b.y(), b.z()};
  double cyz[2] = {c.y(), c.z()};
  double azx[2] = {a.z(), a.x()};
  double bzx[2] = {b.z(), b.x()};
  double czx[2] = {c.z(), c.x()};
  double axy[2] = {a.x(), a.y()};
  double bxy[2] = {b.x(), b.y()};
  double cxy[2] = {c.x(), c.y()};

  return std::sqrt(std::pow(_orient2d(ayz, byz, cyz), 2)
                 + std::pow(_orient2d(azx, bzx, czx), 2)
                 + std::pow(_orient2d(axy, bxy, cxy), 2));
}
//-----------------------------------------------------------------------------
