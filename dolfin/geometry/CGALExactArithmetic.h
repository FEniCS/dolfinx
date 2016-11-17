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
// Last changed: 2016-11-17
//
// Developer note:
//
// This file contains reference implementations of collision detection
// algorithms using exact arithmetic with CGAL. It is not included in
// a normal build but is used as a reference for verification and
// debugging of the inexact DOLFIN collision detection algorithms.
// Enable by setting the flag
// DOLFIN_ENABLE_CGAL_EXACT_ARITHMETIC. This flag should not be
// defined by default.

#ifndef __CGAL_EXACT_ARITHMETIC_H
#define __CGAL_EXACT_ARITHMETIC_H

// Define or undefine this flag for enabling or disabling CGAL and
// exact arithmetic:

#define DOLFIN_ENABLE_CGAL_EXACT_ARITHMETIC 1

#ifndef DOLFIN_ENABLE_CGAL_EXACT_ARITHMETIC

// Comparison macro just bypasses CGAL and test when not enabled
#define CHECK_CGAL(RESULT_DOLFIN, RESULT_CGAL) RESULT_DOLFIN

#else

#define CGAL_CHECK_TOLERANCE 1e-10

#include "Point.h"
#include "predicates.h"
#include <dolfin/log/log.h>
#include <dolfin/math/basic.h>
#include <vector>
#include <algorithm>
#include <sstream>


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

  inline double volume(std::vector<Point> s)
  {
    // Compute volume of simplex s

    if (s.size() < 3)
      return 0;

    else if (s.size() == 3)
    {
      return 0.5 * std::abs(orient2d(s[0].coordinates(),
				     s[1].coordinates(),
				     s[2].coordinates()));
    }
    else if (s.size() == 4)
    {
      return std::abs(orient3d(s[0].coordinates(),
			       s[1].coordinates(),
			       s[2].coordinates(),
			       s[3].coordinates())) / 6.;
    }
    else {

      dolfin_error("CGALExactArithmetic.h",
		   "volume",
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
		   /* "Error in function %s\n DOLFIN: %d\n CGAL: %d", */
		   /* function.c_str(), */
                   /* result_dolfin.size(), */
                   /* result_cgal.size()); */
		   "Error in function %s\n DOLFIN: %s\n CGAL: %s",
		   function.c_str(),
		   s_dolfin.str().c_str(),
		   s_cgal.str().c_str());
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
#include <CGAL/Point_2.h>
#include <CGAL/Triangle_2.h>
#include <CGAL/Segment_2.h>
#include <CGAL/Point_3.h>
#include <CGAL/Triangle_3.h>
#include <CGAL/Segment_3.h>
#include <CGAL/Tetrahedron_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/intersections.h>
#include <CGAL/intersection_of_Polyhedra_3.h>

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>

namespace
{
  // CGAL typedefs
  /* typedef CGAL::Quotient<CGAL::MP_Float> ExactNumber; */
  /* typedef CGAL::Cartesian<ExactNumber>   ExactKernel; */
  typedef CGAL::Exact_predicates_exact_constructions_kernel ExactKernel;
  typedef ExactKernel::FT ExactNumber;

  typedef ExactKernel::Point_2           Point_2;
  typedef ExactKernel::Triangle_2        Triangle_2;
  typedef ExactKernel::Segment_2         Segment_2;
  typedef ExactKernel::Intersect_2       Intersect_2;
  typedef ExactKernel::Point_3           Point_3;
  typedef ExactKernel::Triangle_3        Triangle_3;
  typedef ExactKernel::Segment_3         Segment_3;
  typedef ExactKernel::Tetrahedron_3     Tetrahedron_3;
  typedef ExactKernel::Intersect_3       Intersect_3;
  typedef CGAL::Polyhedron_3<ExactKernel>      Polyhedron_3;

  //---------------------------------------------------------------------------
  // CGAL utility functions
  //---------------------------------------------------------------------------
  inline Point_2 convert_to_cgal_2d(double a, double b)
  {
    return Point_2(a, b);
  }

  inline Point_3 convert_to_cgal_3d(double a, double b, double c)
  {
    return Point_3(a, b, c);
  }

  inline Point_2 convert_to_cgal_2d(const dolfin::Point& p)
  {
    return Point_2(p[0], p[1]);
  }

  inline Point_3 convert_to_cgal_3d(const dolfin::Point& p)
  {
    return Point_3(p[0], p[1], p[2]);
  }

  inline Segment_2 convert_to_cgal_2d(const dolfin::Point& a,
				      const dolfin::Point& b)
  {
    return Segment_2(convert_to_cgal_2d(a), convert_to_cgal_2d(b));
  }

  inline Segment_3 convert_to_cgal_3d(const dolfin::Point& a,
				      const dolfin::Point& b)
  {
    return Segment_3(convert_to_cgal_3d(a), convert_to_cgal_3d(b));
  }

  inline Triangle_2 convert_to_cgal_2d(const dolfin::Point& a,
				       const dolfin::Point& b,
				       const dolfin::Point& c)
  {
    return Triangle_2(convert_to_cgal_2d(a),
		      convert_to_cgal_2d(b),
		      convert_to_cgal_2d(c));
  }

  inline Triangle_3 convert_to_cgal_3d(const dolfin::Point& a,
				       const dolfin::Point& b,
				       const dolfin::Point& c)
  {
    return Triangle_3(convert_to_cgal_3d(a),
		      convert_to_cgal_3d(b),
		      convert_to_cgal_3d(c));
  }

  inline Tetrahedron_3 convert_to_cgal_3d(const dolfin::Point& a,
					  const dolfin::Point& b,
					  const dolfin::Point& c,
					  const dolfin::Point& d)
  {
    return Tetrahedron_3(convert_to_cgal_3d(a),
			 convert_to_cgal_3d(b),
			 convert_to_cgal_3d(c),
			 convert_to_cgal_3d(d));
  }

  //-----------------------------------------------------------------------------
  inline bool is_degenerate_2d(const dolfin::Point& a,
			       const dolfin::Point& b)
  {
    const Segment_2 s(convert_to_cgal_2d(a), convert_to_cgal_2d(b));
    return s.is_degenerate();
  }

  inline bool is_degenerate_3d(const dolfin::Point& a,
			       const dolfin::Point& b)
  {
    const Segment_3 s(convert_to_cgal_3d(a), convert_to_cgal_3d(b));
    return s.is_degenerate();
  }

  inline bool is_degenerate_2d(const dolfin::Point& a,
			       const dolfin::Point& b,
			       const dolfin::Point& c)
  {
    const Triangle_2 t(convert_to_cgal_2d(a),
		       convert_to_cgal_2d(b),
		       convert_to_cgal_2d(c));
    return t.is_degenerate();
  }

  inline bool is_degenerate_3d(const dolfin::Point& a,
			       const dolfin::Point& b,
			       const dolfin::Point& c)
  {
    const Triangle_3 t(convert_to_cgal_3d(a),
		       convert_to_cgal_3d(b),
		       convert_to_cgal_3d(c));
    return t.is_degenerate();
  }

  inline bool is_degenerate_3d(const dolfin::Point& a,
			       const dolfin::Point& b,
			       const dolfin::Point& c,
			       const dolfin::Point& d)
  {
    const Tetrahedron_3 t(convert_to_cgal_3d(a),
			  convert_to_cgal_3d(b),
			  convert_to_cgal_3d(c),
			  convert_to_cgal_3d(d));
    return t.is_degenerate();
  }

  //-----------------------------------------------------------------------------
  inline dolfin::Point convert_from_cgal(const Point_2& p)
  {
    return dolfin::Point(CGAL::to_double(p.x()),CGAL::to_double(p.y()));
  }

  inline dolfin::Point convert_from_cgal(const Point_3& p)
  {
    return dolfin::Point(CGAL::to_double(p.x()),
			 CGAL::to_double(p.y()),
			 CGAL::to_double(p.z()));
  }

  inline std::vector<dolfin::Point> convert_from_cgal(const Segment_2& s)
  {
    const std::vector<dolfin::Point> triangulation =
      {{ dolfin::Point(CGAL::to_double(s.vertex(0)[0]),
		       CGAL::to_double(s.vertex(0)[1])),
	 dolfin::Point(CGAL::to_double(s.vertex(1)[0]),
		       CGAL::to_double(s.vertex(1)[1]))
	}};
    return triangulation;
  }

  inline std::vector<dolfin::Point> convert_from_cgal(const Segment_3& s)
  {
    const std::vector<dolfin::Point> triangulation =
      {{ dolfin::Point(CGAL::to_double(s.vertex(0)[0]),
		       CGAL::to_double(s.vertex(0)[1]),
		       CGAL::to_double(s.vertex(0)[2])),
	 dolfin::Point(CGAL::to_double(s.vertex(1)[0]),
		       CGAL::to_double(s.vertex(1)[1]),
		       CGAL::to_double(s.vertex(1)[2]))
	}};
    return triangulation;
  }

  inline std::vector<dolfin::Point> convert_from_cgal(const Triangle_2& t)
  {
    const std::vector<dolfin::Point> triangulation =
      {{ dolfin::Point(CGAL::to_double(t.vertex(0)[0]),
		       CGAL::to_double(t.vertex(0)[1])),
	 dolfin::Point(CGAL::to_double(t.vertex(2)[0]),
		       CGAL::to_double(t.vertex(2)[1])),
	 dolfin::Point(CGAL::to_double(t.vertex(1)[0]),
		       CGAL::to_double(t.vertex(1)[1]))
	}};
    return triangulation;
  }

  inline std::vector<dolfin::Point> convert_from_cgal(const Triangle_3& t)
  {
    const std::vector<dolfin::Point> triangulation =
      {{ dolfin::Point(CGAL::to_double(t.vertex(0)[0]),
		       CGAL::to_double(t.vertex(0)[1]),
		       CGAL::to_double(t.vertex(0)[2])),
	 dolfin::Point(CGAL::to_double(t.vertex(2)[0]),
		       CGAL::to_double(t.vertex(2)[1]),
		       CGAL::to_double(t.vertex(2)[2])),
	 dolfin::Point(CGAL::to_double(t.vertex(1)[0]),
		       CGAL::to_double(t.vertex(1)[1]),
		       CGAL::to_double(t.vertex(1)[2]))
	}};
    return triangulation;
  }

  inline
  std::vector<std::vector<dolfin::Point>>
  triangulate_polygon_2d(const std::vector<dolfin::Point>& points)
  {
    using Point = dolfin::Point;

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

  inline
  std::vector<std::vector<dolfin::Point>>
  triangulate_polygon_3d(const std::vector<dolfin::Point>& points)
  {
    // FIXME
    dolfin::dolfin_error("CGALExactArithmetic.h",
			 "triangulate_polygon_3d",
			 "Not implemented");
    return std::vector<std::vector<dolfin::Point>>();
  }
}

namespace dolfin
{
  //---------------------------------------------------------------------------
  // Reference implementations of DOLFIN collision detection predicates
  // using CGAL exact arithmetic
  // ---------------------------------------------------------------------------

  inline bool cgal_collides_segment_point_2d(const Point& q0,
					     const Point& q1,
					     const Point& p,
					     bool only_interior=false)
  {
    const Point_2 q0_ = convert_to_cgal_2d(q0);
    const Point_2 q1_ = convert_to_cgal_2d(q1);
    const Point_2 p_ = convert_to_cgal_2d(p);

    const bool intersects = CGAL::do_intersect(Segment_2(q0_, q1_), p_);
    return only_interior ? intersects && p_ != q0_ && p_ != q1_ : intersects;
  }

  inline bool cgal_collides_segment_point_3d(const Point& q0,
					     const Point& q1,
					     const Point& p,
					     bool only_interior=false)
  {
    const Point_3 q0_ = convert_to_cgal_3d(q0);
    const Point_3 q1_ = convert_to_cgal_3d(q1);
    const Point_3 p_ = convert_to_cgal_3d(p);

    const Segment_3 segment(q0_, q1_);
    const bool intersects = segment.has_on(p_);
    return only_interior ? intersects && p_ != q0_ && p_ != q1_ : intersects;
  }

  inline bool cgal_collides_segment_segment_2d(const Point& p0,
					       const Point& p1,
					       const Point& q0,
					       const Point& q1)
  {
    return CGAL::do_intersect(convert_to_cgal_2d(p0, p1),
			      convert_to_cgal_2d(q0, q1));
  }

  inline bool cgal_collides_segment_segment_3d(const Point& p0,
					       const Point& p1,
					       const Point& q0,
					       const Point& q1)
  {
    return CGAL::do_intersect(convert_to_cgal_3d(p0, p1),
			      convert_to_cgal_3d(q0, q1));
  }

  inline bool cgal_collides_triangle_point_2d(const Point& p0,
					      const Point& p1,
					      const Point& p2,
					      const Point &point)
  {
    return CGAL::do_intersect(convert_to_cgal_2d(p0, p1, p2),
			      convert_to_cgal_2d(point));
  }

  inline bool cgal_collides_triangle_point_3d(const Point& p0,
					      const Point& p1,
					      const Point& p2,
					      const Point &point)
  {
    const Triangle_3 tri = convert_to_cgal_3d(p0, p1, p2);
    return tri.has_on(convert_to_cgal_3d(point));
  }


  inline bool cgal_collides_triangle_segment_2d(const Point& p0,
						const Point& p1,
						const Point& p2,
						const Point& q0,
						const Point& q1)
  {
    return CGAL::do_intersect(convert_to_cgal_2d(p0, p1, p2),
			      convert_to_cgal_2d(q0, q1));
  }

  inline bool cgal_collides_triangle_segment_3d(const Point& p0,
						const Point& p1,
						const Point& p2,
						const Point& q0,
						const Point& q1)
  {
    return CGAL::do_intersect(convert_to_cgal_3d(p0, p1, p2),
			      convert_to_cgal_3d(q0, q1));
  }


  inline bool cgal_collides_triangle_triangle_2d(const Point& p0,
						 const Point& p1,
						 const Point& p2,
						 const Point& q0,
						 const Point& q1,
						 const Point& q2)
  {
    return CGAL::do_intersect(convert_to_cgal_2d(p0, p1, p2),
			      convert_to_cgal_2d(q0, q1, q2));
  }

  inline bool cgal_collides_triangle_triangle_3d(const Point& p0,
						 const Point& p1,
						 const Point& p2,
						 const Point& q0,
						 const Point& q1,
						 const Point& q2)
  {
    return CGAL::do_intersect(convert_to_cgal_3d(p0, p1, p2),
			      convert_to_cgal_3d(q0, q1, q2));
  }


  inline bool cgal_collides_tetrahedron_point(const Point& p0,
					      const Point& p1,
					      const Point& p2,
					      const Point& p3,
					      const Point& q0)
  {
    const Tetrahedron_3 tet = convert_to_cgal_3d(p0, p1, p2, p3);
    return !tet.has_on_unbounded_side(convert_to_cgal_3d(q0));
  }

  inline bool cgal_collides_tetrahedron_segment(const Point& p0,
						const Point& p1,
						const Point& p2,
						const Point& p3,
						const Point& q0,
						const Point& q1)
  {
    if (cgal_collides_tetrahedron_point(p0, p1, p2, p3, q0) or
	cgal_collides_tetrahedron_point(p0, p1, p2, p3, q1))
      return true;

    if (cgal_collides_triangle_segment_3d(p0, p1, p2, q0, q1) or
	cgal_collides_triangle_segment_3d(p0, p2, p3, q0, q1) or
	cgal_collides_triangle_segment_3d(p0, p3, p1, q0, q1) or
	cgal_collides_triangle_segment_3d(p1, p3, p2, q0, q1))
      return true;

    return false;
  }

  inline bool cgal_collides_tetrahedron_triangle(const Point& p0,
						 const Point& p1,
						 const Point& p2,
						 const Point& p3,
						 const Point& q0,
						 const Point& q1,
						 const Point& q2)
  {
    return CGAL::do_intersect(convert_to_cgal_3d(p0, p1, p2, p3),
			      convert_to_cgal_3d(q0, q1, q2));
  }

  inline bool cgal_collides_tetrahedron_tetrahedron(const Point& p0,
						    const Point& p1,
						    const Point& p2,
						    const Point& p3,
						    const Point& q0,
						    const Point& q1,
						    const Point& q2,
						    const Point& q3)
  {
    // Check volume collisions
    if (cgal_collides_tetrahedron_point(p0, p1, p2, p3, q0)) return true;
    if (cgal_collides_tetrahedron_point(p0, p1, p2, p3, q1)) return true;
    if (cgal_collides_tetrahedron_point(p0, p1, p2, p3, q2)) return true;
    if (cgal_collides_tetrahedron_point(p0, p1, p2, p3, q3)) return true;
    if (cgal_collides_tetrahedron_point(q0, q1, q2, q3, p0)) return true;
    if (cgal_collides_tetrahedron_point(q0, q1, q2, q3, p1)) return true;
    if (cgal_collides_tetrahedron_point(q0, q1, q2, q3, p2)) return true;
    if (cgal_collides_tetrahedron_point(q0, q1, q2, q3, p3)) return true;

    Polyhedron_3 tet_a;
    tet_a.make_tetrahedron(convert_to_cgal_3d(p0),
			   convert_to_cgal_3d(p1),
			   convert_to_cgal_3d(p2),
			   convert_to_cgal_3d(p3));

    Polyhedron_3 tet_b;
    tet_b.make_tetrahedron(convert_to_cgal_3d(q0),
			   convert_to_cgal_3d(q1),
			   convert_to_cgal_3d(q2),
			   convert_to_cgal_3d(q3));

    // Check for polyhedron intersection (recall that a polyhedron is
    // only its vertices, edges and faces)
    std::size_t cnt = 0;
    CGAL::Counting_output_iterator out(&cnt);
    CGAL::intersection_Polyhedron_3_Polyhedron_3<Polyhedron_3>(tet_a,
							       tet_b,
							       out);
    // The tetrahedra does not intersect if cnt == 0
    return cnt != 0;
  }

  //----------------------------------------------------------------------------
  // Reference implementations of DOLFIN intersection triangulation
  // functions using CGAL exact arithmetic
  // ---------------------------------------------------------------------------
  inline
  std::vector<Point> cgal_intersection_segment_segment_2d(const Point& p0,
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

  inline
  std::vector<Point> cgal_intersection_segment_segment_3d(const Point& p0,
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

  inline
  std::vector<Point> cgal_triangulate_segment_segment_2d(const Point& p0,
							 const Point& p1,
							 const Point& q0,
							 const Point& q1)
  {
    return cgal_intersection_segment_segment_2d(p0, p1, q0, q1);
  }

  //-----------------------------------------------------------------------------
  inline
  std::vector<Point> cgal_triangulate_segment_segment_3d(const Point& p0,
							 const Point& p1,
							 const Point& q0,
							 const Point& q1)
  {
    return cgal_intersection_segment_segment_3d(p0, p1, q0, q1);
  }

  inline
  std::vector<Point>
  cgal_intersection_segment_interior_segment_interior_2d(const Point& p0,
							 const Point& p1,
							 const Point& q0,
							 const Point& q1)
  {
    dolfin_assert(!is_degenerate_2d(p0, p1));
    dolfin_assert(!is_degenerate_2d(q0, q1));

    const Segment_2 I0 = convert_to_cgal_2d(p0, p1);
    const Segment_2 I1 = convert_to_cgal_2d(q0, q1);
    std::vector<Point> triangulation;

    if (const auto ii = CGAL::intersection(I0, I1))
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
      else
      {
	dolfin_error("CGALExactArithmetic.h",
		     "cgal_intersection_segment_interior_segment_interior_2d",
		     "Unexpected behavior");
      }
    }

    return triangulation;
  }

  inline
  std::vector<Point>
  cgal_intersection_segment_interior_segment_interior_3d(const Point& p0,
							 const Point& p1,
							 const Point& q0,
							 const Point& q1)
  {
    dolfin_assert(!is_degenerate_3d(p0, p1));
    dolfin_assert(!is_degenerate_3d(q0, q1));

    const Segment_3 I0 = convert_to_cgal_3d(p0, p1);
    const Segment_3 I1 = convert_to_cgal_3d(q0, q1);
    std::vector<Point> triangulation;

    if (const auto ii = CGAL::intersection(I0, I1))
    {
      if (const Point_3* p = boost::get<Point_3>(&*ii))
      {
	if (*p != I0.source() && *p != I0.target() && *p != I1.source() && *p != I1.target())
	  triangulation.push_back(convert_from_cgal(*p));
      }
      else if (const Segment_3* s = boost::get<Segment_3>(&*ii))
      {
	if (s->source() != I0.source() && s->source() != I0.target() && s->source() != I1.source() && s->source() != I1.target())
	  triangulation.push_back(convert_from_cgal(s->source()));
	if (s->target() != I0.source() && s->target() != I0.target() && s->target() != I1.source() && s->target() != I1.target())
	  triangulation.push_back(convert_from_cgal(s->target()));
      }
      else
      {
	dolfin_error("CGALExactArithmetic.h",
		     "cgal_intersection_segment_interior_segment_interior_3d",
		     "Unexpected behavior");
      }
    }

    return triangulation;
  }

  inline
  std::vector<Point>
  cgal_triangulate_segment_interior_segment_interior_2d(const Point& p0,
							const Point& p1,
							const Point& q0,
							const Point& q1)
  {
    return cgal_intersection_segment_interior_segment_interior_2d(p0, p1, q0, q1);
  }

  inline
  std::vector<Point>
  cgal_triangulate_segment_interior_segment_interior_3d(const Point& p0,
							const Point& p1,
							const Point& q0,
							const Point& q1)
  {
    return cgal_intersection_segment_interior_segment_interior_3d(p0, p1, q0, q1);
  }

  //-----------------------------------------------------------------------------
  inline
  std::vector<Point> cgal_intersection_triangle_segment_2d(const Point& p0,
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

  inline
  std::vector<Point> cgal_intersection_triangle_segment_3d(const Point& p0,
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
		     "cgal_intersection_triangle_segment_3d",
		     "Unexpected behavior");
      }
    }
    return std::vector<Point>();
  }

  //-----------------------------------------------------------------------------
  inline
  std::vector<Point> cgal_triangulate_triangle_segment_2d(const Point& p0,
                                                          const Point& p1,
                                                          const Point& p2,
                                                          const Point& q0,
							  const Point& q1)
  {
    return cgal_intersection_triangle_segment_2d(p0, p1, p2, q0, q1);
  }

  //-----------------------------------------------------------------------------
  inline
  std::vector<Point> cgal_triangulate_triangle_segment_3d(const Point& p0,
							  const Point& p1,
							  const Point& p2,
							  const Point& q0,
							  const Point& q1)
  {
    return cgal_intersection_triangle_segment_3d(p0, p1, p2, q0, q1);
  }
  //-----------------------------------------------------------------------------
  inline
  std::vector<Point> cgal_intersection_triangle_triangle_2d(const Point& p0,
                                                            const Point& p1,
                                                            const Point& p2,
                                                            const Point& q0,
                                                            const Point& q1,
                                                            const Point& q2)
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
        std::cout << "CGAL: Intersection is point" << std::endl;
        intersection.push_back(convert_from_cgal(*p));;
      }
      else if (const Segment_2* s = boost::get<Segment_2>(&*ii))
      {
        std::cout << "CGAL: Intersection is segment: (" << s->source() << ", " << s->target() << ")" << std::endl;
        intersection = convert_from_cgal(*s);
      }
      else if (const Triangle_2* t = boost::get<Triangle_2>(&*ii))
      {
        std::cout << "CGAL: Intersection is triangle" << std::endl;
        std::cout << "Area: " << t->area() << std::endl;
        intersection = convert_from_cgal(*t);;
      }
      else if (const std::vector<Point_2>* cgal_points = boost::get<std::vector<Point_2>>(&*ii))
      {
        std::cout << "CGAL: Intersection is polygon (" << cgal_points->size() << ")" << std::endl;
        for (Point_2 p : *cgal_points)
        {
          intersection.push_back(convert_from_cgal(p));
          std::cout << p << ", ";
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
      /*                "find intersection of two triangles in cgal_intersection_triangle_triangle function", */
      /*                "no intersection found"); */
    }

    return intersection;
  }

  inline
  std::vector<Point> cgal_intersection_triangle_triangle_3d(const Point& p0,
							    const Point& p1,
							    const Point& p2,
							    const Point& q0,
							    const Point& q1,
							    const Point& q2)
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
	std::cout << "CGAL: Intersection is point" << std::endl;
	intersection.push_back(convert_from_cgal(*p));;
      }
      else if (const Segment_3* s = boost::get<Segment_3>(&*ii))
      {
	std::cout << "CGAL: Intersection is segment: (" << s->source() << ", " << s->target() << ")" << std::endl;
	intersection = convert_from_cgal(*s);
      }
      else if (const Triangle_3* t = boost::get<Triangle_3>(&*ii))
      {
	std::cout << "CGAL: Intersection is triangle" << std::endl;
	intersection = convert_from_cgal(*t);;
      }
      else if (const std::vector<Point_3>* cgal_points = boost::get<std::vector<Point_3>>(&*ii))
      {
	std::cout << "CGAL: Intersection is polygon (" << cgal_points->size() << ")" << std::endl;
	for (Point_3 p : *cgal_points)
	{
	  intersection.push_back(convert_from_cgal(p));
	  std::cout << p << ", ";
	}
      }
      else
      {
	dolfin_error("CGALExactArithmetic.h",
		     "cgal_intersection_triangle_triangle_3d",
		     "Unexpected behavior");
      }

      // NB: the parsing can return triangulation of size 0, for example
      // if it detected a triangle but it was found to be flat.
      /* if (triangulation.size() == 0) */
      /*   dolfin_error("CGALExactArithmetic.h", */
      /*                "find intersection of two triangles in cgal_intersection_triangle_triangle function", */
      /*                "no intersection found"); */
    }

    return intersection;
  }

  //----------------------------------------------------------------------------
  inline
  std::vector<std::vector<Point>>
  cgal_triangulate_triangle_triangle_2d(const Point& p0,
					const Point& p1,
					const Point& p2,
					const Point& q0,
					const Point& q1,
					const Point& q2)
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
      dolfin_assert(intersection.size() == 4 ||
		    intersection.size() == 5 ||
		    intersection.size() == 6);
      return triangulate_polygon_2d(intersection);
    }
  }

  inline
  std::vector<std::vector<Point>>
  cgal_triangulate_triangle_triangle_3d(const Point& p0,
					const Point& p1,
					const Point& p2,
					const Point& q0,
					const Point& q1,
					const Point& q2)
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
      dolfin_assert(intersection.size() == 4 ||
		    intersection.size() == 5 ||
		    intersection.size() == 6);
      return triangulate_polygon_3d(intersection);
    }
  }

  //-----------------------------------------------------------------------------
  inline
  std::vector<Point>
  cgal_intersection_tetrahedron_triangle(const Point& p0,
					 const Point& p1,
					 const Point& p2,
					 const Point& p3,
					 const Point& q0,
					 const Point& q1,
					 const Point& q2)
  {
    dolfin_assert(!is_degenerate_3d(p0, p1, p2, p3));
    dolfin_assert(!is_degenerate_3d(q0, q1, q2));

    // const Tetrahedron_3 tet = convert_from_cgal(p0, p1, p2, p3);
    // const Triangle_3 tri = convert_from_cgal(q0, q1, q2);

    Polyhedron_3 tet;
    tet.make_tetrahedron(convert_to_cgal_3d(p0),
    			 convert_to_cgal_3d(p1),
    			 convert_to_cgal_3d(p2),
    			 convert_to_cgal_3d(p3));
    Polyhedron_3 tri;
    tri.make_triangle(convert_to_cgal_3d(q0),
    		      convert_to_cgal_3d(q1),
    		      convert_to_cgal_3d(q2));

    std::list<std::vector<Point_3> > triangulation;
    CGAL::intersection_Polyhedron_3_Polyhedron_3(tet,
    						 tri,
    						 std::back_inserter(triangulation));

    // FIXME: do we need to add interior point checks? Maybe
    // Polyhedron_3 is only top dim 2?

    dolfin_error("CGALExactArithmetic.h",
		 "cgal_intersection_tetrahedron_triangle",
		 "Not implemented");


    //return triangulation;
    return std::vector<Point>();
  }

  //-----------------------------------------------------------------------------
  inline
  std::vector<std::vector<Point>>
  cgal_triangulate_tetrahedron_triangle(const Point& p0,
					const Point& p1,
					const Point& p2,
					const Point& p3,
					const Point& q0,
					const Point& q1,
					const Point& q2)
  {
    dolfin_assert(!is_degenerate_3d(p0, p1, p2, p3));
    dolfin_assert(!is_degenerate_3d(q0, q1, q2));

    std::vector<Point> intersection =
      cgal_intersection_tetrahedron_triangle(p0, p1, p2, p3, q0, q1, q2);

    // FIXME
    dolfin_error("CGALExactArithmetic.h",
		 "cgal_triangulation_tetrahedron_triangle",
		 "Not implemented");

    return std::vector<std::vector<Point>>();
  }

  //-----------------------------------------------------------------------------
  inline
  std::vector<Point>
  cgal_intersection_tetrahedron_tetrahedron(const Point& p0,
					    const Point& p1,
					    const Point& p2,
					    const Point& p3,
					    const Point& q0,
					    const Point& q1,
					    const Point& q2,
					    const Point& q3)
  {
    dolfin_assert(!is_degenerate_3d(p0, p1, p2, p3));
    dolfin_assert(!is_degenerate_3d(q0, q1, q2, q3));

    Polyhedron_3 tet_a;
    tet_a.make_tetrahedron(convert_to_cgal_3d(p0),
			   convert_to_cgal_3d(p1),
			   convert_to_cgal_3d(p2),
			   convert_to_cgal_3d(p3));
    Polyhedron_3 tet_b;
    tet_b.make_tetrahedron(convert_to_cgal_3d(q0),
			   convert_to_cgal_3d(q1),
			   convert_to_cgal_3d(q2),
			   convert_to_cgal_3d(q3));

    std::list<std::vector<Point_3> > triangulation;
    CGAL::intersection_Polyhedron_3_Polyhedron_3(tet_a,
    						 tet_b,
    						 std::back_inserter(triangulation));

    // FIXME: do we need to add interior point checks? Maybe
    // Polyhedron_3 is only top dim 2?


    // FIXME
    dolfin_error("CGALExactArithmetic.h",
		 "cgal_intersection_tetrahedron_tetrahedron",
		 "Not implemented");
    return std::vector<Point>();
  }

  //-----------------------------------------------------------------------------
  inline
  std::vector<std::vector<Point>>
  cgal_triangulate_tetrahedron_tetrahedron(const Point& p0,
					   const Point& p1,
					   const Point& p2,
					   const Point& p3,
					   const Point& q0,
					   const Point& q1,
					   const Point& q2,
					   const Point& q3)
  {
    dolfin_assert(!is_degenerate_3d(p0, p1, p2, p3));
    dolfin_assert(!is_degenerate_3d(q0, q1, q2, q3));

    std::vector<Point> intersection =
      cgal_intersection_tetrahedron_tetrahedron(p0, p1, p2, p3,
						q0, q1, q2, q3);

    // FIXME
    dolfin_error("CGALExactArithmetic.h",
		 "cgal_intersection_tetrahedron_tetrahedron",
		 "Not implemented");

    return std::vector<std::vector<Point>>();
  }

  //----------------------------------------------------------------------------
  // Reference implementations of DOLFIN is_degenerate
  // ---------------------------------------------------------------------------
  inline bool cgal_is_degenerate_2d(const std::vector<Point>& s)
  {
    switch (s.size())
    {
    case 1: return true;
    case 2: return is_degenerate_2d(s[0], s[1]);
    case 3: return is_degenerate_2d(s[0], s[1], s[2]);
    default:
      dolfin_error("CGALExactArithmetic.h",
		   "cgal_is_degenerate_2d",
		   "Only implemented for simplices of tdim 0, 1 and 2");
    }
    return false;
  }

  inline bool cgal_is_degenerate_3d(const std::vector<Point>& s)
  {
    switch (s.size())
    {
    case 1: return true;
    case 2: return is_degenerate_3d(s[0], s[1]);
    case 3: return is_degenerate_3d(s[0], s[1], s[2]);
    case 4: return is_degenerate_3d(s[0], s[1], s[2], s[3]);
    default:
      dolfin_error("CGALExactArithmetic.h",
		   "cgal_is_degenerate_3d",
		   "Only implemented for simplices of tdim 0, 1, 2 and 3");
    }
    return false;
  }

}
#endif

#endif
