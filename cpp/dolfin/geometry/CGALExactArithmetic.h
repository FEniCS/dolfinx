// Copyright (C) 2016-2017 Benjamin Kehlet, August Johansson, and Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifndef DOLFIN_ENABLE_GEOMETRY_DEBUGGING

// Comparison macro just bypasses CGAL and test when not enabled
#define CHECK_CGAL(RESULT_DOLFIN, RESULT_CGAL) RESULT_DOLFIN

#define CGAL_INTERSECTION_CHECK(RESULT_DOLFIN, RESULT_CGAL) RESULT_DOLFIN

#else

#define CGAL_CHECK_TOLERANCE 1e-10

#include "Point.h"
#include "predicates.h"
#include <algorithm>
#include <dolfin/log/log.h>
#include <dolfin/math/basic.h>
#include <iomanip>
#include <sstream>
#include <vector>

// Check that results from DOLFIN and CGAL match
namespace dolfin
{
namespace geometry
{

//---------------------------------------------------------------------------
// Functions to compare results between DOLFIN and CGAL
//---------------------------------------------------------------------------
inline bool check_cgal(bool result_dolfin, bool result_cgal,
                       const std::string& function)
{
  if (result_dolfin != result_cgal)
  {
    // Convert results to strings
    std::stringstream s_dolfin;
    std::stringstream s_cgal;
    s_dolfin << result_dolfin;
    s_cgal << result_cgal;

    // Issue error
    dolfin_error(
        "CGALExactArithmetic.h", "verify geometric predicate with exact types",
        "Error in predicate %s\n DOLFIN: %s\n CGAL: %s", function.c_str(),
        s_dolfin.str().c_str(), s_cgal.str().c_str());
  }

  return result_dolfin;
}
//-----------------------------------------------------------------------------
inline std::vector<Point>
cgal_intersection_check(const std::vector<Point>& dolfin_result,
                        const std::vector<Point>& cgal_result,
                        const std::string& function)
{
  if (dolfin_result.size() != cgal_result.size())
  {
    dolfin_error("CGALExactArithmetic.h", "verify intersection",
                 "size of point set differs (%d vs %d)", dolfin_result.size(),
                 cgal_result.size());
  }

  for (const Point& p1 : dolfin_result)
  {
    bool found = false;
    for (const Point& p2 : cgal_result)
    {
      if ((p1 - p2).norm() < 1e-15)
      {
        found = true;
        break;
      }
    }

    if (!found)
      dolfin_error(
          "CGALExactArithmetic.h", "verify intersection construction result",
          "Point (%f, %f, %f) in dolfin result not found in cgal result", p1[0],
          p1[1], p1[2]);
  }
  return dolfin_result;
}
} // end namespace dolfin
//-----------------------------------------------------------------------------
// Comparison macro that calls comparison function
#define CHECK_CGAL(RESULT_DOLFIN, RESULT_CGAL)                                 \
  check_cgal(RESULT_DOLFIN, RESULT_CGAL, __FUNCTION__)

#define CGAL_INTERSECTION_CHECK(RESULT_DOLFIN, RESULT_CGAL)                    \
  cgal_intersection_check(RESULT_DOLFIN, RESULT_CGAL, __FUNCTION__)

// CGAL includes
#define CGAL_HEADER_ONLY
#include <CGAL/Cartesian.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/MP_Float.h>
#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/Point_2.h>
#include <CGAL/Point_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Quotient.h>
#include <CGAL/Segment_2.h>
#include <CGAL/Segment_3.h>
#include <CGAL/Tetrahedron_3.h>
#include <CGAL/Triangle_2.h>
#include <CGAL/Triangle_3.h>
#include <CGAL/Triangulation_2.h>
#include <CGAL/convex_hull_3.h>
#include <CGAL/intersection_of_Polyhedra_3.h>
#include <CGAL/intersections.h>

namespace
{
// CGAL typedefs
/* typedef CGAL::Quotient<CGAL::MP_Float> ExactNumber; */
/* typedef CGAL::Cartesian<ExactNumber>   ExactKernel; */
typedef CGAL::Exact_predicates_exact_constructions_kernel ExactKernel;
typedef ExactKernel::FT ExactNumber;

typedef ExactKernel::Point_2 Point_2;
typedef ExactKernel::Triangle_2 Triangle_2;
typedef ExactKernel::Segment_2 Segment_2;
typedef ExactKernel::Intersect_2 Intersect_2;
typedef ExactKernel::Point_3 Point_3;
typedef ExactKernel::Vector_3 Vector_3;
typedef ExactKernel::Triangle_3 Triangle_3;
typedef ExactKernel::Segment_3 Segment_3;
typedef ExactKernel::Tetrahedron_3 Tetrahedron_3;
typedef ExactKernel::Intersect_3 Intersect_3;
typedef CGAL::Nef_polyhedron_3<ExactKernel> Nef_polyhedron_3;
typedef CGAL::Polyhedron_3<ExactKernel> Polyhedron_3;
typedef CGAL::Triangulation_2<ExactKernel> Triangulation_2;

//---------------------------------------------------------------------------
// CGAL utility functions
//---------------------------------------------------------------------------
inline Point_2 convert_to_cgal_2d(double a, double b) { return Point_2(a, b); }
//-----------------------------------------------------------------------------
inline Point_3 convert_to_cgal_3d(double a, double b, double c)
{
  return Point_3(a, b, c);
}
//-----------------------------------------------------------------------------
inline Point_2 convert_to_cgal_2d(const dolfin::Point& p)
{
  return Point_2(p[0], p[1]);
}
//-----------------------------------------------------------------------------
inline Point_3 convert_to_cgal_3d(const dolfin::Point& p)
{
  return Point_3(p[0], p[1], p[2]);
}
//-----------------------------------------------------------------------------
inline Segment_2 convert_to_cgal_2d(const dolfin::Point& a,
                                    const dolfin::Point& b)
{
  return Segment_2(convert_to_cgal_2d(a), convert_to_cgal_2d(b));
}
//-----------------------------------------------------------------------------
inline Segment_3 convert_to_cgal_3d(const dolfin::Point& a,
                                    const dolfin::Point& b)
{
  return Segment_3(convert_to_cgal_3d(a), convert_to_cgal_3d(b));
}
//-----------------------------------------------------------------------------
inline Triangle_2 convert_to_cgal_2d(const dolfin::Point& a,
                                     const dolfin::Point& b,
                                     const dolfin::Point& c)
{
  return Triangle_2(convert_to_cgal_2d(a), convert_to_cgal_2d(b),
                    convert_to_cgal_2d(c));
}
//-----------------------------------------------------------------------------
inline Triangle_3 convert_to_cgal_3d(const dolfin::Point& a,
                                     const dolfin::Point& b,
                                     const dolfin::Point& c)
{
  return Triangle_3(convert_to_cgal_3d(a), convert_to_cgal_3d(b),
                    convert_to_cgal_3d(c));
}
//-----------------------------------------------------------------------------
inline Tetrahedron_3 convert_to_cgal_3d(const dolfin::Point& a,
                                        const dolfin::Point& b,
                                        const dolfin::Point& c,
                                        const dolfin::Point& d)
{
  return Tetrahedron_3(convert_to_cgal_3d(a), convert_to_cgal_3d(b),
                       convert_to_cgal_3d(c), convert_to_cgal_3d(d));
}
//-----------------------------------------------------------------------------
inline bool is_degenerate_2d(const dolfin::Point& a, const dolfin::Point& b)
{
  const Segment_2 s(convert_to_cgal_2d(a), convert_to_cgal_2d(b));
  return s.is_degenerate();
}
//-----------------------------------------------------------------------------
inline bool is_degenerate_3d(const dolfin::Point& a, const dolfin::Point& b)
{
  const Segment_3 s(convert_to_cgal_3d(a), convert_to_cgal_3d(b));
  return s.is_degenerate();
}
//-----------------------------------------------------------------------------
inline bool is_degenerate_2d(const dolfin::Point& a, const dolfin::Point& b,
                             const dolfin::Point& c)
{
  const Triangle_2 t(convert_to_cgal_2d(a), convert_to_cgal_2d(b),
                     convert_to_cgal_2d(c));
  return t.is_degenerate();
}
//-----------------------------------------------------------------------------
inline bool is_degenerate_3d(const dolfin::Point& a, const dolfin::Point& b,
                             const dolfin::Point& c)
{
  const Triangle_3 t(convert_to_cgal_3d(a), convert_to_cgal_3d(b),
                     convert_to_cgal_3d(c));
  return t.is_degenerate();
}
//-----------------------------------------------------------------------------
inline bool is_degenerate_3d(const dolfin::Point& a, const dolfin::Point& b,
                             const dolfin::Point& c, const dolfin::Point& d)
{
  const Tetrahedron_3 t(convert_to_cgal_3d(a), convert_to_cgal_3d(b),
                        convert_to_cgal_3d(c), convert_to_cgal_3d(d));
  return t.is_degenerate();
}
//-----------------------------------------------------------------------------
inline dolfin::Point convert_from_cgal(const Point_2& p)
{
  return dolfin::Point(CGAL::to_double(p.x()), CGAL::to_double(p.y()));
}
//-----------------------------------------------------------------------------
inline dolfin::Point convert_from_cgal(const Point_3& p)
{
  return dolfin::Point(CGAL::to_double(p.x()), CGAL::to_double(p.y()),
                       CGAL::to_double(p.z()));
}
//-----------------------------------------------------------------------------
inline std::vector<dolfin::Point> convert_from_cgal(const Segment_2& s)
{
  const std::vector<dolfin::Point> triangulation
      = {{dolfin::Point(CGAL::to_double(s.vertex(0)[0]),
                        CGAL::to_double(s.vertex(0)[1])),
          dolfin::Point(CGAL::to_double(s.vertex(1)[0]),
                        CGAL::to_double(s.vertex(1)[1]))}};
  return triangulation;
}
//-----------------------------------------------------------------------------
inline std::vector<dolfin::Point> convert_from_cgal(const Segment_3& s)
{
  const std::vector<dolfin::Point> triangulation
      = {{dolfin::Point(CGAL::to_double(s.vertex(0)[0]),
                        CGAL::to_double(s.vertex(0)[1]),
                        CGAL::to_double(s.vertex(0)[2])),
          dolfin::Point(CGAL::to_double(s.vertex(1)[0]),
                        CGAL::to_double(s.vertex(1)[1]),
                        CGAL::to_double(s.vertex(1)[2]))}};
  return triangulation;
}
//-----------------------------------------------------------------------------
inline std::vector<dolfin::Point> convert_from_cgal(const Triangle_2& t)
{
  const std::vector<dolfin::Point> triangulation
      = {{dolfin::Point(CGAL::to_double(t.vertex(0)[0]),
                        CGAL::to_double(t.vertex(0)[1])),
          dolfin::Point(CGAL::to_double(t.vertex(2)[0]),
                        CGAL::to_double(t.vertex(2)[1])),
          dolfin::Point(CGAL::to_double(t.vertex(1)[0]),
                        CGAL::to_double(t.vertex(1)[1]))}};
  return triangulation;
}
//-----------------------------------------------------------------------------
inline std::vector<dolfin::Point> convert_from_cgal(const Triangle_3& t)
{
  const std::vector<dolfin::Point> triangulation
      = {{dolfin::Point(CGAL::to_double(t.vertex(0)[0]),
                        CGAL::to_double(t.vertex(0)[1]),
                        CGAL::to_double(t.vertex(0)[2])),
          dolfin::Point(CGAL::to_double(t.vertex(2)[0]),
                        CGAL::to_double(t.vertex(2)[1]),
                        CGAL::to_double(t.vertex(2)[2])),
          dolfin::Point(CGAL::to_double(t.vertex(1)[0]),
                        CGAL::to_double(t.vertex(1)[1]),
                        CGAL::to_double(t.vertex(1)[2]))}};
  return triangulation;
}
//-----------------------------------------------------------------------------
inline std::vector<std::vector<dolfin::Point>>
triangulate_polygon_2d(const std::vector<dolfin::Point>& points)
{
  // Convert points
  std::vector<Point_2> pcgal(points.size());
  for (std::size_t i = 0; i < points.size(); ++i)
    pcgal[i] = convert_to_cgal_2d(points[i]);

  // Triangulate
  Triangulation_2 tcgal;
  tcgal.insert(pcgal.begin(), pcgal.end());

  // Convert back
  std::vector<std::vector<dolfin::Point>> t;
  for (Triangulation_2::Finite_faces_iterator fit = tcgal.finite_faces_begin();
       fit != tcgal.finite_faces_end(); ++fit)
  {
    t.push_back({{convert_from_cgal(tcgal.triangle(fit)[0]),
                  convert_from_cgal(tcgal.triangle(fit)[1]),
                  convert_from_cgal(tcgal.triangle(fit)[2])}});
  }

  return t;
}
//-----------------------------------------------------------------------------
inline std::vector<std::vector<dolfin::Point>>
triangulate_polygon_3d(const std::vector<dolfin::Point>& points)
{
  // FIXME
  dolfin::dolfin_error("CGALExactArithmetic.h", "triangulate_polygon_3d",
                       "Not implemented");
  return std::vector<std::vector<dolfin::Point>>();
}
//-----------------------------------------------------------------------------
}

namespace dolfin
{
//---------------------------------------------------------------------------
// Reference implementations of DOLFIN collision detection predicates
// using CGAL exact arithmetic
// ---------------------------------------------------------------------------
inline bool cgal_collides_segment_point_2d(const Point& q0, const Point& q1,
                                           const Point& p,
                                           bool only_interior = false)
{
  const Point_2 q0_ = convert_to_cgal_2d(q0);
  const Point_2 q1_ = convert_to_cgal_2d(q1);
  const Point_2 p_ = convert_to_cgal_2d(p);

  const bool intersects = CGAL::do_intersect(Segment_2(q0_, q1_), p_);
  return only_interior ? intersects && p_ != q0_ && p_ != q1_ : intersects;
}
//-----------------------------------------------------------------------------
inline bool cgal_collides_segment_point_3d(const Point& q0, const Point& q1,
                                           const Point& p,
                                           bool only_interior = false)
{
  const Point_3 q0_ = convert_to_cgal_3d(q0);
  const Point_3 q1_ = convert_to_cgal_3d(q1);
  const Point_3 p_ = convert_to_cgal_3d(p);

  const Segment_3 segment(q0_, q1_);
  const bool intersects = segment.has_on(p_);
  return only_interior ? intersects && p_ != q0_ && p_ != q1_ : intersects;
}
//-----------------------------------------------------------------------------
inline bool cgal_collides_segment_segment_2d(const Point& p0, const Point& p1,
                                             const Point& q0, const Point& q1)
{
  return CGAL::do_intersect(convert_to_cgal_2d(p0, p1),
                            convert_to_cgal_2d(q0, q1));
}
//-----------------------------------------------------------------------------
inline bool cgal_collides_segment_segment_3d(const Point& p0, const Point& p1,
                                             const Point& q0, const Point& q1)
{
  return CGAL::do_intersect(convert_to_cgal_3d(p0, p1),
                            convert_to_cgal_3d(q0, q1));
}
//-----------------------------------------------------------------------------
inline bool cgal_collides_triangle_point_2d(const Point& p0, const Point& p1,
                                            const Point& p2, const Point& point)
{
  return CGAL::do_intersect(convert_to_cgal_2d(p0, p1, p2),
                            convert_to_cgal_2d(point));
}
//-----------------------------------------------------------------------------
inline bool cgal_collides_triangle_point_3d(const Point& p0, const Point& p1,
                                            const Point& p2, const Point& point)
{
  const Triangle_3 tri = convert_to_cgal_3d(p0, p1, p2);
  return tri.has_on(convert_to_cgal_3d(point));
}
//-----------------------------------------------------------------------------
inline bool cgal_collides_triangle_segment_2d(const Point& p0, const Point& p1,
                                              const Point& p2, const Point& q0,
                                              const Point& q1)
{
  return CGAL::do_intersect(convert_to_cgal_2d(p0, p1, p2),
                            convert_to_cgal_2d(q0, q1));
}
//-----------------------------------------------------------------------------
inline bool cgal_collides_triangle_segment_3d(const Point& p0, const Point& p1,
                                              const Point& p2, const Point& q0,
                                              const Point& q1)
{
  return CGAL::do_intersect(convert_to_cgal_3d(p0, p1, p2),
                            convert_to_cgal_3d(q0, q1));
}
//-----------------------------------------------------------------------------
inline bool cgal_collides_triangle_triangle_2d(const Point& p0, const Point& p1,
                                               const Point& p2, const Point& q0,
                                               const Point& q1, const Point& q2)
{
  return CGAL::do_intersect(convert_to_cgal_2d(p0, p1, p2),
                            convert_to_cgal_2d(q0, q1, q2));
}
//-----------------------------------------------------------------------------
inline bool cgal_collides_triangle_triangle_3d(const Point& p0, const Point& p1,
                                               const Point& p2, const Point& q0,
                                               const Point& q1, const Point& q2)
{
  return CGAL::do_intersect(convert_to_cgal_3d(p0, p1, p2),
                            convert_to_cgal_3d(q0, q1, q2));
}
//-----------------------------------------------------------------------------
inline bool cgal_collides_tetrahedron_point_3d(const Point& p0, const Point& p1,
                                               const Point& p2, const Point& p3,
                                               const Point& q0)
{
  const Tetrahedron_3 tet = convert_to_cgal_3d(p0, p1, p2, p3);
  return !tet.has_on_unbounded_side(convert_to_cgal_3d(q0));
}
//-----------------------------------------------------------------------------
inline bool
cgal_collides_tetrahedron_segment_3d(const Point& p0, const Point& p1,
                                     const Point& p2, const Point& p3,
                                     const Point& q0, const Point& q1)
{
  if (cgal_collides_tetrahedron_point_3d(p0, p1, p2, p3, q0)
      or cgal_collides_tetrahedron_point_3d(p0, p1, p2, p3, q1))
    return true;

  if (cgal_collides_triangle_segment_3d(p0, p1, p2, q0, q1)
      or cgal_collides_triangle_segment_3d(p0, p2, p3, q0, q1)
      or cgal_collides_triangle_segment_3d(p0, p3, p1, q0, q1)
      or cgal_collides_triangle_segment_3d(p1, p3, p2, q0, q1))
    return true;

  return false;
}
//-----------------------------------------------------------------------------
inline bool cgal_collides_tetrahedron_triangle_3d(
    const Point& p0, const Point& p1, const Point& p2, const Point& p3,
    const Point& q0, const Point& q1, const Point& q2)
{
  return CGAL::do_intersect(convert_to_cgal_3d(p0, p1, p2, p3),
                            convert_to_cgal_3d(q0, q1, q2));
}
//-----------------------------------------------------------------------------
inline bool cgal_collides_tetrahedron_tetrahedron_3d(
    const Point& p0, const Point& p1, const Point& p2, const Point& p3,
    const Point& q0, const Point& q1, const Point& q2, const Point& q3)
{
  // Check volume collisions
  if (cgal_collides_tetrahedron_point_3d(p0, p1, p2, p3, q0))
    return true;
  if (cgal_collides_tetrahedron_point_3d(p0, p1, p2, p3, q1))
    return true;
  if (cgal_collides_tetrahedron_point_3d(p0, p1, p2, p3, q2))
    return true;
  if (cgal_collides_tetrahedron_point_3d(p0, p1, p2, p3, q3))
    return true;
  if (cgal_collides_tetrahedron_point_3d(q0, q1, q2, q3, p0))
    return true;
  if (cgal_collides_tetrahedron_point_3d(q0, q1, q2, q3, p1))
    return true;
  if (cgal_collides_tetrahedron_point_3d(q0, q1, q2, q3, p2))
    return true;
  if (cgal_collides_tetrahedron_point_3d(q0, q1, q2, q3, p3))
    return true;

  Polyhedron_3 tet_a;
  tet_a.make_tetrahedron(convert_to_cgal_3d(p0), convert_to_cgal_3d(p1),
                         convert_to_cgal_3d(p2), convert_to_cgal_3d(p3));

  Polyhedron_3 tet_b;
  tet_b.make_tetrahedron(convert_to_cgal_3d(q0), convert_to_cgal_3d(q1),
                         convert_to_cgal_3d(q2), convert_to_cgal_3d(q3));

  // Check for polyhedron intersection (recall that a polyhedron is
  // only its vertices, edges and faces)
  std::size_t cnt = 0;
  CGAL::Counting_output_iterator out(&cnt);
  CGAL::intersection_Polyhedron_3_Polyhedron_3<Polyhedron_3>(tet_a, tet_b, out);
  // The tetrahedra does not intersect if cnt == 0
  return cnt != 0;
}
//----------------------------------------------------------------------------
// Reference implementations of DOLFIN intersection triangulation
// functions using CGAL with exact arithmetic
// ---------------------------------------------------------------------------
inline std::vector<Point> cgal_intersection_segment_segment_2d(const Point& p0,
                                                               const Point& p1,
                                                               const Point& q0,
                                                               const Point& q1)
{
  dolfin_assert(!is_degenerate_2d(p0, p1));
  dolfin_assert(!is_degenerate_2d(q0, q1));

  const auto I0 = convert_to_cgal_2d(p0, p1);
  const auto I1 = convert_to_cgal_2d(q0, q1);

  if (const auto ii = CGAL::intersection(I0, I1))
  {
    if (const Point_2* p = boost::get<Point_2>(&*ii))
    {
      return std::vector<Point>{convert_from_cgal(*p)};
    }
    else if (const Segment_2* s = boost::get<Segment_2>(&*ii))
    {
      return convert_from_cgal(*s);
    }
    else
    {
      dolfin_error("CGALExactArithmetic.h",
                   "cgal_intersection_segment_segment_2d",
                   "Unexpected behavior");
    }
  }

  return std::vector<Point>();
}
//-----------------------------------------------------------------------------
inline std::vector<Point> cgal_intersection_segment_segment_3d(const Point& p0,
                                                               const Point& p1,
                                                               const Point& q0,
                                                               const Point& q1)
{
  dolfin_assert(!is_degenerate_3d(p0, p1));
  dolfin_assert(!is_degenerate_3d(q0, q1));

  const auto I0 = convert_to_cgal_3d(p0, p1);
  const auto I1 = convert_to_cgal_3d(q0, q1);

  if (const auto ii = CGAL::intersection(I0, I1))
  {
    if (const Point_3* p = boost::get<Point_3>(&*ii))
    {
      return std::vector<Point>{convert_from_cgal(*p)};
    }
    else if (const Segment_3* s = boost::get<Segment_3>(&*ii))
    {
      return convert_from_cgal(*s);
    }
    else
    {
      dolfin_error("CGALExactArithmetic.h",
                   "cgal_intersection_segment_segment_3d",
                   "Unexpected behavior");
    }
  }

  return std::vector<Point>();
}
//-----------------------------------------------------------------------------
inline std::vector<Point> cgal_triangulate_segment_segment_2d(const Point& p0,
                                                              const Point& p1,
                                                              const Point& q0,
                                                              const Point& q1)
{
  return cgal_intersection_segment_segment_2d(p0, p1, q0, q1);
}
//-----------------------------------------------------------------------------
inline std::vector<Point> cgal_triangulate_segment_segment_3d(const Point& p0,
                                                              const Point& p1,
                                                              const Point& q0,
                                                              const Point& q1)
{
  return cgal_intersection_segment_segment_3d(p0, p1, q0, q1);
}
//-----------------------------------------------------------------------------
inline std::vector<Point> cgal_intersection_triangle_segment_2d(const Point& p0,
                                                                const Point& p1,
                                                                const Point& p2,
                                                                const Point& q0,
                                                                const Point& q1)
{
  dolfin_assert(!is_degenerate_2d(p0, p1, p2));
  dolfin_assert(!is_degenerate_2d(q0, q1));

  const auto T = convert_to_cgal_2d(p0, p1, p2);
  const auto I = convert_to_cgal_2d(q0, q1);

  if (const auto ii = CGAL::intersection(T, I))
  {
    if (const Point_2* p = boost::get<Point_2>(&*ii))
    {
      return std::vector<Point>{convert_from_cgal(*p)};
    }
    else if (const Segment_2* s = boost::get<Segment_2>(&*ii))
    {
      return convert_from_cgal(*s);
    }
    else
    {
      dolfin_error("CGALExactArithmetic.h",
                   "cgal_intersection_triangle_segment_2d",
                   "Unexpected behavior");
    }
  }

  return std::vector<Point>();
}
//-----------------------------------------------------------------------------
inline std::vector<Point> cgal_intersection_triangle_segment_3d(const Point& p0,
                                                                const Point& p1,
                                                                const Point& p2,
                                                                const Point& q0,
                                                                const Point& q1)
{
  dolfin_assert(!is_degenerate_3d(p0, p1, p2));
  dolfin_assert(!is_degenerate_3d(q0, q1));

  const auto T = convert_to_cgal_3d(p0, p1, p2);
  const auto I = convert_to_cgal_3d(q0, q1);

  if (const auto ii = CGAL::intersection(T, I))
  {
    if (const Point_3* p = boost::get<Point_3>(&*ii))
      return std::vector<Point>{convert_from_cgal(*p)};
    else if (const Segment_3* s = boost::get<Segment_3>(&*ii))
      return convert_from_cgal(*s);
    else
    {
      dolfin_error("CGALExactArithmetic.h",
                   "cgal_intersection_triangle_segment_3d",
                   "Unexpected behavior");
    }
  }

  return std::vector<Point>();
}
//-----------------------------------------------------------------------------
inline std::vector<Point> cgal_triangulate_triangle_segment_2d(const Point& p0,
                                                               const Point& p1,
                                                               const Point& p2,
                                                               const Point& q0,
                                                               const Point& q1)
{
  return cgal_intersection_triangle_segment_2d(p0, p1, p2, q0, q1);
}
//-----------------------------------------------------------------------------
inline std::vector<Point> cgal_triangulate_triangle_segment_3d(const Point& p0,
                                                               const Point& p1,
                                                               const Point& p2,
                                                               const Point& q0,
                                                               const Point& q1)
{
  return cgal_intersection_triangle_segment_3d(p0, p1, p2, q0, q1);
}
//-----------------------------------------------------------------------------
inline std::vector<Point>
cgal_intersection_triangle_triangle_2d(const Point& p0, const Point& p1,
                                       const Point& p2, const Point& q0,
                                       const Point& q1, const Point& q2)
{
  dolfin_assert(!is_degenerate_2d(p0, p1, p2));
  dolfin_assert(!is_degenerate_2d(q0, q1, q2));

  const Triangle_2 T0 = convert_to_cgal_2d(p0, p1, p2);
  const Triangle_2 T1 = convert_to_cgal_2d(q0, q1, q2);
  std::vector<Point> intersection;

  if (const auto ii = CGAL::intersection(T0, T1))
  {
    if (const Point_2* p = boost::get<Point_2>(&*ii))
    {
      intersection.push_back(convert_from_cgal(*p));
    }
    else if (const Segment_2* s = boost::get<Segment_2>(&*ii))
    {
      intersection = convert_from_cgal(*s);
    }
    else if (const Triangle_2* t = boost::get<Triangle_2>(&*ii))
    {
      intersection = convert_from_cgal(*t);
      ;
    }
    else if (const std::vector<Point_2>* cgal_points
             = boost::get<std::vector<Point_2>>(&*ii))
    {
      for (Point_2 p : *cgal_points)
      {
        intersection.push_back(convert_from_cgal(p));
      }
    }
    else
    {
      dolfin_error("CGALExactArithmetic.h",
                   "cgal_intersection_triangle_triangle_2d",
                   "Unexpected behavior");
    }

    // NB: the parsing can return triangulation of size 0, for example
    // if it detected a triangle but it was found to be flat.
    /* if (triangulation.size() == 0) */
    /*   dolfin_error("CGALExactArithmetic.h", */
    /*                "find intersection of two triangles in
     * cgal_intersection_triangle_triangle function", */
    /*                "no intersection found"); */
  }

  return intersection;
}
//-----------------------------------------------------------------------------
inline std::vector<Point>
cgal_intersection_triangle_triangle_3d(const Point& p0, const Point& p1,
                                       const Point& p2, const Point& q0,
                                       const Point& q1, const Point& q2)
{
  dolfin_assert(!is_degenerate_3d(p0, p1, p2));
  dolfin_assert(!is_degenerate_3d(q0, q1, q2));

  const Triangle_3 T0 = convert_to_cgal_3d(p0, p1, p2);
  const Triangle_3 T1 = convert_to_cgal_3d(q0, q1, q2);
  std::vector<Point> intersection;

  if (const auto ii = CGAL::intersection(T0, T1))
  {
    if (const Point_3* p = boost::get<Point_3>(&*ii))
    {
      intersection.push_back(convert_from_cgal(*p));
    }
    else if (const Segment_3* s = boost::get<Segment_3>(&*ii))
    {
      intersection = convert_from_cgal(*s);
    }
    else if (const Triangle_3* t = boost::get<Triangle_3>(&*ii))
    {
      intersection = convert_from_cgal(*t);
      ;
    }
    else if (const std::vector<Point_3>* cgal_points
             = boost::get<std::vector<Point_3>>(&*ii))
    {
      for (Point_3 p : *cgal_points)
      {
        intersection.push_back(convert_from_cgal(p));
      }
    }
    else
    {
      dolfin_error("CGALExactArithmetic.h",
                   "cgal_intersection_triangle_triangle_3d",
                   "Unexpected behavior");
    }
  }

  return intersection;
}
//----------------------------------------------------------------------------
inline std::vector<std::vector<Point>>
cgal_triangulate_triangle_triangle_2d(const Point& p0, const Point& p1,
                                      const Point& p2, const Point& q0,
                                      const Point& q1, const Point& q2)
{
  dolfin_assert(!is_degenerate_2d(p0, p1, p2));
  dolfin_assert(!is_degenerate_2d(q0, q1, q2));

  const std::vector<Point> intersection
      = cgal_intersection_triangle_triangle_2d(p0, p1, p2, q0, q1, q2);

  if (intersection.size() < 4)
  {
    return std::vector<std::vector<Point>>{intersection};
  }
  else
  {
    dolfin_assert(intersection.size() == 4 || intersection.size() == 5
                  || intersection.size() == 6);
    return triangulate_polygon_2d(intersection);
  }
}
//-----------------------------------------------------------------------------
inline std::vector<std::vector<Point>>
cgal_triangulate_triangle_triangle_3d(const Point& p0, const Point& p1,
                                      const Point& p2, const Point& q0,
                                      const Point& q1, const Point& q2)
{
  dolfin_assert(!is_degenerate_3d(p0, p1, p2));
  dolfin_assert(!is_degenerate_3d(q0, q1, q2));

  const std::vector<Point> intersection
      = cgal_intersection_triangle_triangle_3d(p0, p1, p2, q0, q1, q2);

  if (intersection.size() < 4)
  {
    return std::vector<std::vector<Point>>{intersection};
  }
  else
  {
    dolfin_assert(intersection.size() == 4 || intersection.size() == 5
                  || intersection.size() == 6);
    return triangulate_polygon_3d(intersection);
  }
}
//-----------------------------------------------------------------------------
inline std::vector<Point> cgal_intersection_tetrahedron_triangle(
    const Point& p0, const Point& p1, const Point& p2, const Point& p3,
    const Point& q0, const Point& q1, const Point& q2)
{
  dolfin_assert(!is_degenerate_3d(p0, p1, p2, p3));
  dolfin_assert(!is_degenerate_3d(q0, q1, q2));

  // const Tetrahedron_3 tet = convert_from_cgal(p0, p1, p2, p3);
  // const Triangle_3 tri = convert_from_cgal(q0, q1, q2);

  Polyhedron_3 tet;
  tet.make_tetrahedron(convert_to_cgal_3d(p0), convert_to_cgal_3d(p1),
                       convert_to_cgal_3d(p2), convert_to_cgal_3d(p3));
  Polyhedron_3 tri;
  tri.make_triangle(convert_to_cgal_3d(q0), convert_to_cgal_3d(q1),
                    convert_to_cgal_3d(q2));

  std::list<std::vector<Point_3>> triangulation;
  CGAL::intersection_Polyhedron_3_Polyhedron_3(
      tet, tri, std::back_inserter(triangulation));

  // FIXME: do we need to add interior point checks? Maybe
  // Polyhedron_3 is only top dim 2?

  // Shouldn't get here
  dolfin_error("CGALExactArithmetic.h",
               "cgal_intersection_tetrahedron_triangle", "Not implemented");

  return std::vector<Point>();
}
//-----------------------------------------------------------------------------
inline std::vector<std::vector<Point>> cgal_triangulate_tetrahedron_triangle(
    const Point& p0, const Point& p1, const Point& p2, const Point& p3,
    const Point& q0, const Point& q1, const Point& q2)
{
  dolfin_assert(!is_degenerate_3d(p0, p1, p2, p3));
  dolfin_assert(!is_degenerate_3d(q0, q1, q2));

  std::vector<Point> intersection
      = cgal_intersection_tetrahedron_triangle(p0, p1, p2, p3, q0, q1, q2);

  // Shouldn't get here
  dolfin_error("CGALExactArithmetic.h",
               "cgal_triangulation_tetrahedron_triangle", "Not implemented");

  return std::vector<std::vector<Point>>();
}
//-----------------------------------------------------------------------------
inline std::vector<Point> cgal_intersection_tetrahedron_tetrahedron_3d(
    const Point& p0, const Point& p1, const Point& p2, const Point& p3,
    const Point& q0, const Point& q1, const Point& q2, const Point& q3)
{
  dolfin_assert(!is_degenerate_3d(p0, p1, p2, p3));
  dolfin_assert(!is_degenerate_3d(q0, q1, q2, q3));

  Polyhedron_3 tet_a;
  tet_a.make_tetrahedron(convert_to_cgal_3d(p0), convert_to_cgal_3d(p1),
                         convert_to_cgal_3d(p2), convert_to_cgal_3d(p3));
  Polyhedron_3 tet_b;
  tet_b.make_tetrahedron(convert_to_cgal_3d(q0), convert_to_cgal_3d(q1),
                         convert_to_cgal_3d(q2), convert_to_cgal_3d(q3));

  const Nef_polyhedron_3 tet_a_nef(tet_a);
  const Nef_polyhedron_3 tet_b_nef(tet_b);

  const Nef_polyhedron_3 intersection_nef = tet_a_nef * tet_a_nef;

  Polyhedron_3 intersection;
  intersection_nef.convert_to_polyhedron(intersection);

  std::vector<Point> res;
  for (Polyhedron_3::Vertex_const_iterator vit = intersection.vertices_begin();
       vit != intersection.vertices_end(); vit++)
  {
    res.push_back(Point(CGAL::to_double(vit->point().x()),
                        CGAL::to_double(vit->point().y()),
                        CGAL::to_double(vit->point().z())));
  }

  return res;
}
//----------------------------------------------------------------------------
// Reference implementations of DOLFIN is_degenerate
//-----------------------------------------------------------------------------
inline bool cgal_is_degenerate_2d(const std::vector<Point>& s)
{
  if (s.size() < 2 or s.size() > 3)
  {
    info("Degenerate 2D simplex with %d vertices.", s.size());
    return true;
  }

  switch (s.size())
  {
  case 2:
    return is_degenerate_2d(s[0], s[1]);
  case 3:
    return is_degenerate_2d(s[0], s[1], s[2]);
  }

  // Shouldn't get here
  dolfin_error(
      "CGALExactArithmetic.h", "call cgal_is_degenerate_2d",
      "Only implemented for simplices of tdim 0, 1 and 2, not tdim = %d",
      s.size() - 1);

  return true;
}
//-----------------------------------------------------------------------------
inline bool cgal_is_degenerate_3d(const std::vector<Point>& s)
{
  if (s.size() < 2 or s.size() > 4)
  {
    info("Degenerate 3D simplex with %d vertices.", s.size());
    return true;
  }

  switch (s.size())
  {
  case 2:
    return is_degenerate_3d(s[0], s[1]);
  case 3:
    return is_degenerate_3d(s[0], s[1], s[2]);
  case 4:
    return is_degenerate_3d(s[0], s[1], s[2], s[3]);
  }

  // Shouldn't get here
  dolfin_error(
      "CGALExactArithmetic.h", "call cgal_is_degenerate_3d",
      "Only implemented for simplices of tdim 0, 1, 2 and 3, not tdim = %d",
      s.size() - 1);

  return true;
}
//-----------------------------------------------------------------------------
// Computes the volume of the convex hull of the given points
inline double cgal_polyhedron_volume(const std::vector<Point>& ch)
{
  std::vector<Point_3> exact_points;
  exact_points.reserve(ch.size());
  for (const Point p : ch)
  {
    exact_points.push_back(Point_3(p.x(), p.y(), p.z()));
  }

  // Compute the convex hull as a cgal polyhedron_3
  Polyhedron_3 p;
  CGAL::convex_hull_3(exact_points.begin(), exact_points.end(), p);

  ExactNumber volume = .0;
  for (Polyhedron_3::Facet_const_iterator it = p.facets_begin();
       it != p.facets_end(); it++)
  {
    const Polyhedron_3::Halfedge_const_handle h = it->halfedge();
    const Vector_3 V0 = h->vertex()->point() - CGAL::ORIGIN;
    const Vector_3 V1 = h->next()->vertex()->point() - CGAL::ORIGIN;
    const Vector_3 V2 = h->next()->next()->vertex()->point() - CGAL::ORIGIN;

    volume += V0 * CGAL::cross_product(V1, V2);
  }

  return std::abs(CGAL::to_double(volume / 6.0));
}
//-----------------------------------------------------------------------------
inline double cgal_tet_volume(const std::vector<Point>& ch)
{
  dolfin_assert(ch.size() == 3);
  return CGAL::to_double(
      CGAL::volume(Point_3(ch[0].x(), ch[0].y(), ch[0].z()),
                   Point_3(ch[1].x(), ch[1].y(), ch[1].z()),
                   Point_3(ch[2].x(), ch[2].y(), ch[2].z()),
                   Point_3(ch[3].x(), ch[3].y(), ch[3].z())));
}
//-----------------------------------------------------------------------------
inline bool cgal_tet_is_degenerate(const std::vector<Point>& t)
{
  Tetrahedron_3 tet(Point_3(t[0].x(), t[0].y(), t[0].z()),
                    Point_3(t[1].x(), t[1].y(), t[1].z()),
                    Point_3(t[2].x(), t[2].y(), t[2].z()),
                    Point_3(t[3].x(), t[3].y(), t[3].z()));

  return tet.is_degenerate();
}

//-----------------------------------------------------------------------------
inline bool
cgal_triangulation_has_degenerate(std::vector<std::vector<Point>> triangulation)
{
  for (const std::vector<Point>& t : triangulation)
  {
    if (cgal_tet_is_degenerate(t))
      return true;
  }

  return false;
}

//-----------------------------------------------------------------------------
inline bool
cgal_triangulation_overlap(std::vector<std::vector<Point>> triangulation)
{
  std::vector<Tetrahedron_3> tets;

  for (const std::vector<Point>& t : triangulation)
  {
    Tetrahedron_3 tet(Point_3(t[0].x(), t[0].y(), t[0].z()),
                      Point_3(t[1].x(), t[1].y(), t[1].z()),
                      Point_3(t[2].x(), t[2].y(), t[2].z()),
                      Point_3(t[3].x(), t[3].y(), t[3].z()));

    for (const Tetrahedron_3& t0 : tets)
    {
      for (int i = 0; i < 4; i++)
      {
        if (t0.has_on_bounded_side(tet[i]) || tet.has_on_bounded_side(t0[i]))
          return true;
      }
    }

    tets.push_back(tet);
  }

  return false;
}

//-----------------------------------------------------------------------------
}
}
#endif
