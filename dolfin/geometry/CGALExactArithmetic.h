// Copyright (C) 2016 Benjamin Kehlet, August Johansson, and Anders Logg
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
// First added:  2016-05-03
// Last changed: 2016-05-03
//
// Developer note:
//
// This file contains reference implementations of collision detection
// algorithms using exact arithmetic with CGAL. It is not included in
// a normal build but is used as a reference for verification and
// debugging of the inexact DOLFIN collision detection algorithms.
// Enable by setting the flag DOLFIN_ENABLE_CGAL_EXACT_ARITHMETIC.

#ifndef __CGAL_EXACT_ARITHMETIC_H
#define __CGAL_EXACT_ARITHMETIC_H

// FIXME: Debugging
#define DOLFIN_ENABLE_CGAL_EXACT_ARITHMETIC 1

#ifndef DOLFIN_ENABLE_CGAL_EXACT_ARITHMETIC

// Comparison macro just bypasses CGAL and test when not enabled
#define CHECK_CGAL(RESULT_DOLFIN, RESULT_CGAL) RESULT_DOLFIN

#else

// DOLFIN includes
#include <sstream>
#include <dolfin/log/log.h>
#include "IntersectionTriangulation.h"

// Check that results from DOLFIN and CGAL match
namespace dolfin
{
  template<typename T> bool check_cgal(T result_dolfin, T result_cgal,
                                       std::string function)
  {
    if (result_dolfin != result_cgal)
    {
      // Convert results to strings
      std::stringstream s_dolfin;
      std::stringstream s_cgal;
      s_dolfin << result_dolfin;
      s_cgal << result_cgal;

      // Issue error
      dolfin_error("CGALExactArithmetic.cpp",
                   "verify geometric predicate with exact types",
                   "Predicate: %s, DOLFIN: %s CGAL: %s",
                   function.c_str(), s_dolfin.str().c_str(), s_cgal.str().c_str());
    }

    return result_dolfin;
  }
}

// Comparison macro
#define CHECK_CGAL(RESULT_DOLFIN, RESULT_CGAL) check_cgal(RESULT_DOLFIN, RESULT_CGAL, __FUNCTION__)

// CGAL includes
#define CGAL_HEADER_ONLY
#include <CGAL/Cartesian.h>
#include <CGAL/Quotient.h>
#include <CGAL/MP_Float.h>
#include <CGAL/Triangle_2.h>
#include <CGAL/intersection_2.h>

// CGAL typedefs
typedef CGAL::Quotient<CGAL::MP_Float> ExactNumber;
typedef CGAL::Cartesian<ExactNumber>   ExactKernel;
typedef ExactKernel::Point_2           Point_2;
typedef ExactKernel::Triangle_2        Triangle_2;
typedef ExactKernel::Segment_2         Segment_2;

namespace
{
  //---------------------------------------------------------------------------
  // CGAL utility functions
  //---------------------------------------------------------------------------

  inline Point_2 convert_to_cgal(double a, double b)
  {
    return Point_2(a, b);
  }

  inline Point_2 convert_to_cgal(const dolfin::Point& p)
  {
    return Point_2(p[0], p[1]);
  }

  inline Segment_2 convert_to_cgal(const dolfin::Point& a,
                                   const dolfin::Point& b)
  {
    return Segment_2(convert_to_cgal(a), convert_to_cgal(b));
  }

  inline Triangle_2 convert_to_cgal(const dolfin::Point& a,
                                    const dolfin::Point& b,
                                    const dolfin::Point& c)
  {
    return Triangle_2(convert_to_cgal(a), convert_to_cgal(b), convert_to_cgal(c));
  }

  inline std::vector<dolfin::Point> convert_to_cgal(const Segment_2& t)
  {
    std::vector<dolfin::Point> p(2);
    for (std::size_t i = 0; i < 2; ++i)
      for (std::size_t j = 0; j < 2; ++j)
	p[i][j] = CGAL::to_double(t[i][j]);
    return p;
  }

  inline std::vector<dolfin::Point> convert_to_cgal(const Triangle_2& t)
  {
    std::vector<dolfin::Point> p(3);
    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t j = 0; j < 2; ++j)
	p[i][j] = CGAL::to_double(t[i][j]);
    return p;
  }

  inline bool is_degenerate(const dolfin::Point& a,
			    const dolfin::Point& b)
  {
    Segment_2 s(convert_to_cgal(a), convert_to_cgal(b));
    return s.is_degenerate();
  }

  inline bool is_degenerate(const dolfin::Point& a,
			    const dolfin::Point& b,
			    const dolfin::Point& c)
  {
    Triangle_2 t(convert_to_cgal(a), convert_to_cgal(b), convert_to_cgal(c));
    return t.is_degenerate();
  }

  template<class T>
  inline std::vector<double> parse(const T& ii)
  {
    const Point_2* p = boost::get<Point_2>(&*ii);
    if (p)
    {
      std::vector<double> triangulation = {{ CGAL::to_double(p->x()),
					     CGAL::to_double(p->y()) }};
      return triangulation;
    }

    const Segment_2* s = boost::get<Segment_2>(&*ii);
    if (s)
    {
      std::vector<double> triangulation = {{ CGAL::to_double(s->vertex(0)[0]),
    					     CGAL::to_double(s->vertex(0)[1]),
    					     CGAL::to_double(s->vertex(1)[0]),
    					     CGAL::to_double(s->vertex(1)[1]) }};
      return triangulation;
    }

    const Triangle_2* t = boost::get<Triangle_2>(&*ii);
    if (t)
    {
      std::vector<double> triangulation = {{ CGAL::to_double(t->vertex(0)[0]),
    					     CGAL::to_double(t->vertex(0)[1]),
    					     CGAL::to_double(t->vertex(2)[0]),
    					     CGAL::to_double(t->vertex(2)[1]),
    					     CGAL::to_double(t->vertex(1)[0]),
    					     CGAL::to_double(t->vertex(1)[1]) }};
      return triangulation;
    }

    const std::vector<Point_2>* cgal_points = boost::get<std::vector<Point_2>>(&*ii);
    if (cgal_points)
    {
      std::vector<double> triangulation;
      std::vector<dolfin::Point> points(cgal_points->size());
      for (std::size_t i = 0; i < points.size(); ++i)
    	points[i] = dolfin::Point(CGAL::to_double((*cgal_points)[i].x()),
    				  CGAL::to_double((*cgal_points)[i].y()));
      triangulation = dolfin::IntersectionTriangulation::graham_scan(points);
      return triangulation;
    }

    dolfin::error("Unexpected behavior in CGAL tools.");

    return std::vector<double>();
  }
}

namespace dolfin
{
  //---------------------------------------------------------------------------
  // Reference implementations of DOLFIN functions using CGAL exact arithmetic
  //---------------------------------------------------------------------------
  bool cgal_collides_interval_interval(const MeshEntity& interval_0,
                                       const MeshEntity& interval_1)
  {
    const MeshGeometry& geometry_0 = interval_0.mesh().geometry();
    const MeshGeometry& geometry_1 = interval_1.mesh().geometry();
    const unsigned int* vertices_0 = interval_0.entities(0);
    const unsigned int* vertices_1 = interval_1.entities(0);

    const Point a(geometry_0.point(vertices_0[0])[0],
                  geometry_0.point(vertices_0[0])[1]);
    const Point b(geometry_0.point(vertices_0[1])[0],
                  geometry_0.point(vertices_0[1])[1]);
    const Point c(geometry_1.point(vertices_1[0])[0],
                  geometry_1.point(vertices_1[0])[1]);
    const Point d(geometry_1.point(vertices_1[1])[0],
                  geometry_1.point(vertices_1[1])[1]);

    return CGAL::do_intersect(convert_to_cgal(a, b),
                              convert_to_cgal(c, d));
  }

  bool cgal_collides_edge_edge(const Point& a,
			       const Point& b,
			       const Point& c,
			       const Point& d)
  {
    return CGAL::do_intersect(convert_to_cgal(a, b),
			      convert_to_cgal(c, d));
  }

  bool cgal_collides_interval_point(const Point& p0,
				    const Point& p1,
				    const Point& point)
  {
    return CGAL::do_intersect(convert_to_cgal(p0, p1),
			      convert_to_cgal(point));
  }

  bool cgal_collides_triangle_point_2d(const Point& p0,
				       const Point& p1,
				       const Point& p2,
				       const Point &point)
  {
    return CGAL::do_intersect(convert_to_cgal(p0, p1, p2),
			      convert_to_cgal(point));
  }

  bool cgal_collides_triangle_point(const Point& p0,
				    const Point& p1,
				    const Point& p2,
				    const Point &point)
  {
    return CGAL::do_intersect(convert_to_cgal(p0, p1, p2),
			      convert_to_cgal(point));
  }

  bool cgal_collides_triangle_interval(const Point& p0,
				       const Point& p1,
				       const Point& p2,
				       const Point& q0,
				       const Point& q1)
  {
    return CGAL::do_intersect(convert_to_cgal(p0, p1, p2),
			      convert_to_cgal(q0, q1));
  }

  bool cgal_collides_triangle_triangle(const Point& p0,
				       const Point& p1,
				       const Point& p2,
				       const Point& q0,
				       const Point& q1,
				       const Point& q2)
  {
    return CGAL::do_intersect(convert_to_cgal(p0, p1, p2),
			      convert_to_cgal(q0, q1, q2));
  }

  //---------------------------------------------------------------------------
}
#endif

#endif
