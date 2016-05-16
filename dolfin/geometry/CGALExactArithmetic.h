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
// Last changed: 2016-05-09
//
// Developer note:
//
// This file contains reference implementations of collision detection
// algorithms using exact arithmetic with CGAL. It is not included in
// a normal build but is used as a reference for verification and
// debugging of the inexact DOLFIN collision detection algorithms.
// Enable by setting the flag DOLFIN_ENABLE_CGAL_EXACT_ARITHMETIC.

// FIXME: This line is a test - remove it

#ifndef __CGAL_EXACT_ARITHMETIC_H
#define __CGAL_EXACT_ARITHMETIC_H

// FIXME: Debugging
#define DOLFIN_ENABLE_CGAL_EXACT_ARITHMETIC 1

#ifndef DOLFIN_ENABLE_CGAL_EXACT_ARITHMETIC

// Comparison macro just bypasses CGAL and test when not enabled
#define CHECK_CGAL(RESULT_DOLFIN, RESULT_CGAL, __FUNCTION__) RESULT_DOLFIN

#else

#define CGAL_CHECK_TOLERANCE 1e-10

// Includes
#include <algorithm>
#include <sstream>
#include <dolfin/log/log.h>
#include <dolfin/math/basic.h>
#include "Point.h"

// Check that results from DOLFIN and CGAL match
namespace dolfin
{
  //---------------------------------------------------------------------------
  // Functions to compare results between DOLFIN and CGAL
  //---------------------------------------------------------------------------

  inline bool
    check_cgal(bool result_dolfin,
	       bool result_cgal,
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
      dolfin_error("CGALExactArithmetic.h",
                   "verify geometric predicate with exact types",
                   "Error in predicate %s\n DOLFIN: %s\n CGAL: %s",
                   function.c_str(), s_dolfin.str().c_str(), s_cgal.str().c_str());
    }

    return result_dolfin;
  }


  inline double Heron(double a, double b, double c)
  {
    // sort
    if (b>a) std::swap(b,a);
    if (c>b) std::swap(c,b);
    if (b>a) std::swap(b,a);

    const double s2 = (a+(b+c))*(c-(a-b))*(c+(a-b))*(a+(b-c));
    if (s2 < 0)
    {
      std::cout << "Heron error, negative sqrt: " << s2 << " is to be replaced with 0" << std::endl;
      if (std::abs(s2) < DOLFIN_EPS)
	return 0;
      else
	exit(1);
    }
    return 0.25*std::sqrt( (a+(b+c))*(c-(a-b))*(c+(a-b))*(a+(b-c)) );
  }

  inline double volume(const std::vector<Point>& s)
  {
    if (s.size() < 3)
      return 0;
    else if (s.size() == 3)
      return Heron((s[0] - s[1]).norm(),
		   (s[0] - s[2]).norm(),
		   (s[1] - s[2]).norm());
    else if (s.size() == 4)
    {
      dolfin_error("CGALExactArithmetic.h",
		   "verify geometric predicate with exact types",
		   "Volume of tetrahedron not implemented.");
    }
    else {
      dolfin_error("CGALExactArithmetic.h",
		   "verify geometric predicate with exact types",
		   "Volume of simplex with %s points not implemented.", s.size());
    }
    return 0;
  }

  inline const std::vector<Point>&
    check_cgal(const std::vector<Point>& result_dolfin,
               const std::vector<Point>& result_cgal,
               std::string function)
  {
    // compare volume
    const double dolfin_volume = volume(result_dolfin);
    const double cgal_volume = volume(result_cgal);

    if (std::abs(dolfin_volume - cgal_volume) > CGAL_CHECK_TOLERANCE)
    {
      std::stringstream s_dolfin, s_cgal, s_error;
      s_dolfin.precision(16);
      s_dolfin << dolfin_volume;
      s_cgal.precision(16);
      s_cgal << cgal_volume;
      s_error.precision(16);
      s_error << std::abs(dolfin_volume - cgal_volume);

      dolfin_error("CGALExactArithmetic.h",
		   "verify intersections due to different volumes (single simplex version)",
		   "Error in function %s\n CGAL volume %s\n DOLFIN volume %s\n error %s\n",
		   function.c_str(),
		   s_cgal.str().c_str(),
		   s_dolfin.str().c_str(),
		   s_error.str().c_str());
    }

    return result_dolfin;
  }

  inline const std::vector<std::vector<Point>>&
    check_cgal(const std::vector<std::vector<Point>>& result_dolfin,
	       const std::vector<std::vector<Point>>& result_cgal,
	       std::string function)
  {
    // FIXME: do we expect dolfin and cgal data to be in the same order?

    if (result_dolfin.size() != result_cgal.size())
    {
      std::stringstream s_dolfin;
      s_dolfin.precision(16);
      for (const std::vector<Point> s: result_dolfin)
      {
        s_dolfin << "[";
        for (const Point v : s)
          s_dolfin << v << ' ';
        // s_dolfin.seekp(-1, s_dolfin.cur);
        s_dolfin << "]";
      }

      std::stringstream s_cgal;
      s_cgal.precision(16);
      for (const std::vector<Point> s : result_cgal)
      {
        s_cgal << "[";
        for (const Point v : s)
          s_cgal << v << ' ';
        //s_cgal.seekp(-1, s_cgal.cur);
        s_cgal << "]";
      }

      dolfin_error("CGALExactArithmetic.h",
		   "verify intersections due to different sizes",
		   "Error in function %s\n DOLFIN: %d\n CGAL: %d",
		   function.c_str(),
                   result_dolfin.size(),
                   result_cgal.size());
		   /* s_dolfin.str().c_str(), */
		   /* s_cgal.str().c_str()); */
    }
    else
    {
      // compare total volume
      double dolfin_volume = 0;
      for (std::vector<Point> s : result_dolfin)
	dolfin_volume += volume(s);

      double cgal_volume = 0;
      for (std::vector<Point> s : result_cgal)
	cgal_volume += volume(s);

      if (std::abs(cgal_volume - dolfin_volume) > CGAL_CHECK_TOLERANCE)
      {
	std::stringstream s_dolfin, s_cgal, s_error;
	s_dolfin.precision(16);
	s_dolfin << dolfin_volume;
	s_cgal.precision(16);
	s_cgal << cgal_volume;
	s_error.precision(16);
	s_error << std::abs(cgal_volume - dolfin_volume);

	dolfin_error("CGALExactArithmetic.h",
		     "verify intersections due to different volumes",
		     "Error in function %s\n CGAL volume   %s\n DOLFIN volume %s\n error %s\n",
		     function.c_str(),
		     s_cgal.str().c_str(),
		     s_dolfin.str().c_str(),
		     s_error.str().c_str());
      }
    }

    return result_dolfin;
  }

  inline const Point&
    check_cgal(const Point& result_dolfin,
	       const Point& result_cgal,
	       std::string function)
  {
    for (std::size_t d = 0; d < 3; ++d)
    {
      if (!near(result_dolfin[d], result_cgal[d], CGAL_CHECK_TOLERANCE))
      {
	std::stringstream s_dolfin;
	s_dolfin.precision(16);
	std::stringstream s_cgal;
	s_cgal.precision(16);
	for (std::size_t i = 0; i < 3; ++i)
	{
	  s_dolfin << result_dolfin[i] << " ";
	  s_cgal << result_cgal[i] << " ";
	}

	dolfin_error("CGALExactArithmetic.h",
		     "verify intersections due to different Point data",
		     "Error in function %s\n DOLFIN: %s\n CGAL: %s",
		     function.c_str(),
		     s_dolfin.str().c_str(),
		     s_cgal.str().c_str());
      }
    }

    return result_dolfin;
  }

} // end namespace dolfin

// Comparison macro that calls comparison function
#define CHECK_CGAL(RESULT_DOLFIN, RESULT_CGAL) check_cgal(RESULT_DOLFIN, RESULT_CGAL, __FUNCTION__)

// CGAL includes
#define CGAL_HEADER_ONLY
#include <CGAL/Cartesian.h>
#include <CGAL/Quotient.h>
#include <CGAL/MP_Float.h>
#include <CGAL/Triangle_2.h>
#include <CGAL/intersection_2.h>

namespace
{
  // CGAL typedefs
  typedef CGAL::Quotient<CGAL::MP_Float> ExactNumber;
  typedef CGAL::Cartesian<ExactNumber>   ExactKernel;
  typedef ExactKernel::Point_2           Point_2;
  typedef ExactKernel::Triangle_2        Triangle_2;
  typedef ExactKernel::Segment_2         Segment_2;
  typedef ExactKernel::Intersect_2       Intersect_2;

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

  inline dolfin::Point convert_from_cgal(const Point_2& p)
  {
    return dolfin::Point(CGAL::to_double(p.x()),CGAL::to_double(p.y()));
  }

  inline std::vector<dolfin::Point> convert_from_cgal(const Segment_2& s)
  {
    const std::vector<dolfin::Point> triangulation = {{ dolfin::Point(CGAL::to_double(s.vertex(0)[0]),
                                                                      CGAL::to_double(s.vertex(0)[1])),
                                                        dolfin::Point(CGAL::to_double(s.vertex(1)[0]),
                                                                      CGAL::to_double(s.vertex(1)[1]))
      }};
    return triangulation;
  }

  inline std::vector<dolfin::Point> convert_from_cgal(const Triangle_2& t)
  {
    const std::vector<dolfin::Point> triangulation = {{ dolfin::Point(CGAL::to_double(t.vertex(0)[0]),
								      CGAL::to_double(t.vertex(0)[1])),
                                                        dolfin::Point(CGAL::to_double(t.vertex(2)[0]),
								      CGAL::to_double(t.vertex(2)[1])),
                                                        dolfin::Point(CGAL::to_double(t.vertex(1)[0]),
								      CGAL::to_double(t.vertex(1)[1]))
      }};
    return triangulation;
  }

  inline std::vector<std::vector<dolfin::Point>> triangulate_polygon(const std::vector<dolfin::Point>& points)
  {
    using Point = dolfin::Point;

    //std::vector<std::vector<Point>>
    // IntersectionTriangulation::graham_scan(const std::vector<Point>& points)
    // {
    // NB: The input points should be unique.

    // Sometimes we can get an extra point on an edge: a-----c--b. This
    // point c may cause problems for the graham scan. To avoid this,
    // use an extra center point.  Use this center point and point no 0
    // as reference for the angle calculation
    Point pointscenter = points[0];
    for (std::size_t m = 1; m < points.size(); ++m)
      pointscenter += points[m];
    pointscenter /= points.size();

  std::vector<std::pair<double, std::size_t>> order;
  Point ref = points[0] - pointscenter;
  ref /= ref.norm();

  // Compute normal
  Point normal = (points[2] - points[0]).cross(points[1] - points[0]);
  const double det = normal.norm();
  normal /= det;

  // Calculate and store angles
  for (std::size_t m = 1; m < points.size(); ++m)
  {
    const Point v = points[m] - pointscenter;
    const double frac = ref.dot(v) / v.norm();
    double alpha;
    if (frac <= -1)
      alpha = DOLFIN_PI;
    else if (frac >= 1)
      alpha = 0;
    else
    {
      alpha = acos(frac);
      if (v.dot(normal.cross(ref)) < 0)
        alpha = 2*DOLFIN_PI-alpha;
    }
    order.push_back(std::make_pair(alpha, m));
  }

  // Sort angles
  std::sort(order.begin(), order.end());

  // Tessellate
  std::vector<std::vector<Point>> triangulation(order.size() - 1);
  for (std::size_t m = 0; m < order.size()-1; ++m)
  {
    // FIXME: We could consider only triangles with area > tolerance here.
    triangulation[m] = {{ points[0],
			  points[order[m].second],
			  points[order[m + 1].second] }};
  }

  return triangulation;
  }

  //------------------------------------------------------------------------------
  // Explicit handling of CGAL intersections
  //------------------------------------------------------------------------------

  inline std::vector<dolfin::Point>
    parse_segment_segment_intersection
    (const CGAL::cpp11::result_of<Intersect_2(Segment_2, Segment_2)>::type ii)
  {
    const Point_2* p = boost::get<Point_2>(&*ii);
    if (p)
      return std::vector<dolfin::Point>{convert_from_cgal(*p)};

    const Segment_2* s = boost::get<Segment_2>(&*ii);
    if (s)
      return convert_from_cgal(*s);

    dolfin::error("Unexpected behavior in CGALExactArithmetic parse_segment_segment");
    return std::vector<dolfin::Point>();
  }

  inline std::vector<dolfin::Point>
    parse_triangle_segment_intersection
    (const CGAL::cpp11::result_of<Intersect_2(Triangle_2, Segment_2)>::type ii)
  {
    const Point_2* p = boost::get<Point_2>(&*ii);
    if (p)
      return std::vector<dolfin::Point>{convert_from_cgal(*p)};

    const Segment_2* s = boost::get<Segment_2>(&*ii);
    if (s)
      return convert_from_cgal(*s);

    dolfin::error("Unexpected behavior in CGALExactArithmetic parse_triangle_segment");
    return std::vector<dolfin::Point>();
  }

  inline std::vector<std::vector<dolfin::Point>>
    parse_triangle_triangle_intersection
    (const CGAL::cpp11::result_of<Intersect_2(Triangle_2, Triangle_2)>::type ii)
  {
    const Point_2* p = boost::get<Point_2>(&*ii);
    if (p)
      return std::vector<std::vector<dolfin::Point>>{{convert_from_cgal(*p)}};

    const Segment_2* s = boost::get<Segment_2>(&*ii);
    if (s)
      return std::vector<std::vector<dolfin::Point>>{{convert_from_cgal(*s)}};

    const Triangle_2* t = boost::get<Triangle_2>(&*ii);
    if (t)
      return std::vector<std::vector<dolfin::Point>>{{convert_from_cgal(*t)}};

    const std::vector<Point_2>* cgal_points = boost::get<std::vector<Point_2>>(&*ii);
    if (cgal_points)
    {
      dolfin_assert(cgal_points->size() == 4);
      return triangulate_polygon(*cgal_points);
    }

    dolfin::error("Unexpected behavior in CGALExactArithmetic parse_triangle_triangle");
    return std::vector<std::vector<dolfin::Point>>();
  }
}

namespace dolfin
{
  //---------------------------------------------------------------------------
  // Reference implementations of DOLFIN collision detection predicates
  // using CGAL exact arithmetic
  // ---------------------------------------------------------------------------

  inline bool cgal_collides_segment_point(const Point& p0,
                                          const Point& p1,
                                          const Point& point)
  {
    return CGAL::do_intersect(convert_to_cgal(p0, p1),
			      convert_to_cgal(point));
  }

  inline bool cgal_collides_segment_segment(const Point& p0,
                                            const Point& p1,
                                            const Point& q0,
                                            const Point& q1)
  {
    return CGAL::do_intersect(convert_to_cgal(p0, p1),
			      convert_to_cgal(q0, q1));
  }


  inline bool cgal_collides_triangle_point(const Point& p0,
					   const Point& p1,
					   const Point& p2,
					   const Point &point)
  {
    return CGAL::do_intersect(convert_to_cgal(p0, p1, p2),
			      convert_to_cgal(point));
  }

  inline bool cgal_collides_triangle_segment(const Point& p0,
                                             const Point& p1,
                                             const Point& p2,
                                             const Point& q0,
                                             const Point& q1)
  {
    return CGAL::do_intersect(convert_to_cgal(p0, p1, p2),
			      convert_to_cgal(q0, q1));
  }

  inline bool cgal_collides_triangle_triangle(const Point& p0,
					      const Point& p1,
					      const Point& p2,
					      const Point& q0,
					      const Point& q1,
					      const Point& q2)
  {
    return CGAL::do_intersect(convert_to_cgal(p0, p1, p2),
			      convert_to_cgal(q0, q1, q2));
  }

  //----------------------------------------------------------------------------
  // Reference implementations of DOLFIN intersection triangulation
  // functions using CGAL exact arithmetic
  // ---------------------------------------------------------------------------

  inline
    std::vector<Point> cgal_triangulate_segment_segment_2d(const Point& p0,
							   const Point& p1,
							   const Point& q0,
							   const Point& q1)
  {
    dolfin_assert(!is_degenerate(p0, p1));
    dolfin_assert(!is_degenerate(q0, q1));

    const auto I0 = convert_to_cgal(p0, p1);
    const auto I1 = convert_to_cgal(q0, q1);
    const auto ii = CGAL::intersection(I0, I1);
    dolfin_assert(ii);
    const std::vector<Point> triangulation = parse_segment_segment_intersection(ii);

    if (triangulation.size() == 0)
      dolfin_error("CGALExactArithmetic.h",
		   "in cgal_intersection_segment_segment function",
		   "unknown intersection");

    return triangulation;
  }

  inline
  std::vector<Point> cgal_triangulate_segment_interior_segment_interior_2d(const Point& p0,
                                                                           const Point& p1,
                                                                           const Point& q0,
                                                                           const Point& q1)
  {
    dolfin_assert(!is_degenerate(p0, p1));
    dolfin_assert(!is_degenerate(q0, q1));

    std::vector<Point> triangulation;

    const Segment_2 I0 = convert_to_cgal(p0, p1);
    const Segment_2 I1 = convert_to_cgal(q0, q1);
    const auto ii = CGAL::intersection(I0, I1);

    if (ii)
    {
      if (const Point_2* p = boost::get<Point_2>(&*ii))
      {
        if (*p != I0.source() && *p != I0.target() && *p != I1.source() && *p != I1.target())
          triangulation.push_back(convert_from_cgal(*p));
      }
      else if (const Segment_2* s = boost::get<Segment_2>(&*ii))
      {
        if (s->source() != I0.source() && s->source() != I0.target() && s->source() != I1.source() && s->source() != I1.target())
          triangulation.push_back(convert_from_cgal(s->source()));
        if (s->target() != I0.source() && s->target() != I0.target() && s->target() != I1.source() && s->target() != I1.target())
          triangulation.push_back(convert_from_cgal(s->target()));

      }
    }
    return triangulation;
  }


  inline
  std::vector<Point> cgal_triangulate_triangle_segment_2d(const Point& p0,
                                                          const Point& p1,
                                                          const Point& p2,
                                                          const Point& q0,
							  const Point& q1)
  {
    dolfin_assert(!is_degenerate(p0, p1, p2));
    dolfin_assert(!is_degenerate(q0, q1));

    const auto T = convert_to_cgal(p0, p1, p2);
    const auto I = convert_to_cgal(q0, q1);
    const auto ii = CGAL::intersection(T, I);
    dolfin_assert(ii);
    const std::vector<Point> triangulation = parse_triangle_segment_intersection(ii);

    if (triangulation.size() == 0)
      dolfin_error("CGALExactArithmetic.h",
		   "in cgal_intersection_triangle_segment function",
		   "unknown intersection");

    return triangulation;
  }

  inline
    std::vector<std::vector<Point>> cgal_triangulate_triangle_triangle_2d(const Point& p0,
									  const Point& p1,
									  const Point& p2,
									  const Point& q0,
									  const Point& q1,
									  const Point& q2)

  {
    dolfin_assert(!is_degenerate(p0, p1, p2));
    dolfin_assert(!is_degenerate(q0, q1, q2));

    const Triangle_2 T0 = convert_to_cgal(p0, p1, p2);
    const Triangle_2 T1 = convert_to_cgal(q0, q1, q2);
    const auto ii = CGAL::intersection(T0, T1);

    // We can have empty ii if we use
    // CGAL::Exact_predicates_inexact_constructions_kernel
    // dolfin_assert(ii);

    std::vector<std::vector<dolfin::Point>> triangulation;

    if (ii)
    {
      if (const Point_2* p = boost::get<Point_2>(&*ii))
      {
        std::cout << "CGAL: Intersection is point" << std::endl;
        triangulation = std::vector<std::vector<dolfin::Point>>{{convert_from_cgal(*p)}};
      }
      else if (const Segment_2* s = boost::get<Segment_2>(&*ii))
      {
        std::cout << "CGAL: Intersection is segment: (" << s->source() << ", " << s->target() << ")" << std::endl;
        triangulation = std::vector<std::vector<dolfin::Point>>{{convert_from_cgal(*s)}};
      }
      else if (const Triangle_2* t = boost::get<Triangle_2>(&*ii))
      {
        std::cout << "CGAL: Intersection is triangle" << std::endl;
        std::cout << "Area: " << t->area() << std::endl;
        triangulation = std::vector<std::vector<dolfin::Point>>{{convert_from_cgal(*t)}};
      }
      else if (const std::vector<Point_2>* cgal_points = boost::get<std::vector<Point_2>>(&*ii))
      {
        std::cout << "CGAL: Intersection is polygon (" << cgal_points->size() << ")" << std::endl;
        std::vector<dolfin::Point> points;
        for (Point_2 p : *cgal_points)
        {
          points.push_back(dolfin::Point(CGAL::to_double(p.x()), CGAL::to_double(p.y())));
          std::cout << p << ", ";
        }
        std::cout << std::endl;
        dolfin_assert(cgal_points->size() == 4 || cgal_points->size() == 5);
        triangulation = triangulate_polygon(points);
      }
      else
      {
        dolfin::error("Unexpected behavior in CGALExactArithmetic parse_triangle_triangle");
        return std::vector<std::vector<dolfin::Point>>();
      }

      // NB: the parsing can return triangulation of size 0, for example
      // if it detected a triangle but it was found to be flat.
      /* if (triangulation.size() == 0) */
      /*   dolfin_error("CGALExactArithmetic.h", */
      /*                "find intersection of two triangles in cgal_intersection_triangle_triangle function", */
      /*                "no intersection found"); */
    }
    else
    {
      std::cout << "CGAL: No intersection" << std::endl;
    }
    return triangulation;
  }
}
#endif

#endif
