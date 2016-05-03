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

#ifndef DOLFIN_ENABLE_CGAL_EXACT_ARITHMETIC

template<typename T>
bool check_cgal(T a, T b)
{
  if (a != b)
    dolfin_error("CGALExactArithmetic.cpp",
                 "verifying geometric predicate with exact types",
                 "Verification failed");
  return a;
}

#define CHECK_CGAL(result, ...) check_cgal(result, __VA_ARGS__)

#else

define CHECK_CGAL(result,

#define CGAL_HEADER_ONLY

#include <CGAL/Cartesian.h>
#include <CGAL/Quotient.h>
#include <CGAL/MP_Float.h>

#include <CGAL/Triangle_2.h>
#include <CGAL/intersection_2.h>


typedef CGAL::Quotient<CGAL::MP_Float>   ExactNumber;
typedef CGAL::Cartesian<ExactNumber>     ExactKernel;
typedef ExactKernel::Point_2             Point_2;
typedef ExactKernel::Triangle_2          Triangle_2;
typedef ExactKernel::Segment_2           Segment_2;

namespace
{
  inline Point_2 convert_to_cgal(double a, double b)
  {
    return Point_2(a, b);
  }

  inline Point_2 convert_to_cgal(const dolfin::Point& p)
  {
    //std::cout << "point convert " << p[0]<<' '<<p[1]<<std::endl;
    return Point_2(p[0], p[1]);
  }

  inline Segment_2 convert_to_cgal(const dolfin::Point& a,
                                   const dolfin::Point& b)
  {
    //std::cout << "segment convert " << std::endl;
    return Segment_2(convert_to_cgal(a), convert(b));
  }

  inline Triangle_2 convert_to_cgal(const dolfin::Point& a,
			    const dolfin::Point& b,
			    const dolfin::Point& c)
  {
    //std::cout << "triangle convert " << std::endl;
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


  // for parsing the intersection, check
  // http://doc.cgal.org/latest/Kernel_23/group__intersection__linear__grp.html
  /* inline std::vector<double> parse(const Point_2& p) */
  /* { */
  /*   std::vector<double> triangulation(2); */
  /*   triangulation[0] = CGAL::to_double(p.x()); */
  /*   triangulation[1] = CGAL::to_double(p.y()); */
  /*   return triangulation; */
  /* } */

  /* inline std::vector<double> parse(const Segment_2& s) */
  /* { */
  /*   std::vector<double> triangulation(4); */
  /*   triangulation[0] = CGAL::to_double(s->vertex(0)[0]); */
  /*   triangulation[1] = CGAL::to_double(s->vertex(0)[1]); */
  /*   triangulation[2] = CGAL::to_double(s->vertex(1)[0]); */
  /*   triangulation[3] = CGAL::to_double(s->vertex(1)[1]); */
  /*   return triangulation; */
  /* } */

  /* inline std::vector<double> parse(const Line_2& l) */
  /* { */
  /*   std::vector<double> triangulation(4); */
  /*   triangulation[0] = CGAL::to_double(l->point(0)[0]); */
  /*   triangulation[1] = CGAL::to_double(l->point(0)[1]); */
  /*   triangulation[2] = CGAL::to_double(l->point(1)[0]); */
  /*   triangulation[3] = CGAL::to_double(l->point(1)[1]); */
  /*   return triangulation; */
  /* } */

  /* inline std::vector<double> parse(const Triangle_2& t) */
  /* { */
  /*   std::vector<double> triangulation(6); */
  /*   triangulation[0] = CGAL::to_double(t->vertex(0)[0]); */
  /*   triangulation[1] = CGAL::to_double(t->vertex(0)[1]); */
  /*   triangulation[2] = CGAL::to_double(t->vertex(1)[0]); */
  /*   triangulation[3] = CGAL::to_double(t->vertex(1)[1]); */
  /*   triangulation[4] = CGAL::to_double(t->vertex(2)[0]); */
  /*   triangulation[5] = CGAL::to_double(t->vertex(2)[1]); */
  /*   return triangulation; */
  /* } */

  /* inline std::vector<double> parse(const std::vector<Point_2>& pts) */
  /* { */
  /*   std::vector<double> triangulation(pts.size()*2); */
  /*   for (std::size_t i = 0; i < pts.size(); ++i) */
  /*   { */
  /*     triangulation[2*i] = CGAL::to_double(pts[i].x()); */
  /*     triangulation[2*i+1] = CGAL::to_double(pts[i].y()); */
  /*   } */
  /*   return triangulation; */
  /* } */


    template<class T>
    inline std::vector<double> parse(const T& ii)
  {
    //std::cout << __FUNCTION__ << std::endl;

    const Point_2* p = boost::get<Point_2>(&*ii);
    if (p)
    {
      //std::cout << "point\n";
      std::vector<double> triangulation = {{ CGAL::to_double(p->x()),
					     CGAL::to_double(p->y()) }};
      return triangulation;
    }

    const Segment_2* s = boost::get<Segment_2>(&*ii);
    if (s)
    {
      //std::cout << "segment " << std::endl;
      //std::cout << (*s)[0][0] <<' '<<(*s)[0][1] <<' '<<(*s)[1][0]<<' '<<(*s)[1][1]<<std::endl;
      std::vector<double> triangulation = {{ CGAL::to_double(s->vertex(0)[0]),
    					     CGAL::to_double(s->vertex(0)[1]),
    					     CGAL::to_double(s->vertex(1)[0]),
    					     CGAL::to_double(s->vertex(1)[1]) }};
      return triangulation;
    }

    const Triangle_2* t = boost::relaxed_get<Triangle_2>(&*ii);
    if (t)
    {
      /* std::cout << "cgal triangle " << std::endl; */
      std::vector<double> triangulation = {{ CGAL::to_double(t->vertex(0)[0]),
    					     CGAL::to_double(t->vertex(0)[1]),
    					     CGAL::to_double(t->vertex(2)[0]),
    					     CGAL::to_double(t->vertex(2)[1]),
    					     CGAL::to_double(t->vertex(1)[0]),
    					     CGAL::to_double(t->vertex(1)[1]) }};
      return triangulation;
    }

    const std::vector<Point_2>* cgal_points = boost::relaxed_get<std::vector<Point_2>>(&*ii);
    if (cgal_points)
    {
      /* std::cout << "cgal triangulation " << std::endl; */
      std::vector<double> triangulation;

      // convert to dolfin::Point
      std::vector<dolfin::Point> points(cgal_points->size());
      for (std::size_t i = 0; i < points.size(); ++i)
    	points[i] = dolfin::Point(CGAL::to_double((*cgal_points)[i].x()),
    				  CGAL::to_double((*cgal_points)[i].y()));

      triangulation = dolfin::IntersectionTriangulation::graham_scan(points);
      return triangulation;

/* #ifdef Augustdebug_cgal */
/*       std::cout << "before duplicates "<< points.size() << '\n'; */
/*       for (std::size_t i = 0; i < points.size(); ++i) */
/* 	std::cout << tools::matlabplot(points[i]); */
/*       std::cout << '\n'; */
/* #endif */

/*       // remove duplicate points */
/*       std::vector<Point> tmp; */
/*       tmp.reserve(points.size()); */

/*       for (std::size_t i = 0; i < points.size(); ++i) */
/*       { */
/* 	bool different = true; */
/* 	for (std::size_t j = i+1; j < points.size(); ++j) */
/* 	  if ((points[i] - points[j]).norm() < DOLFIN_EPS)//_LARGE) */
/* 	  { */
/* 	    different = false; */
/* 	    break; */
/* 	  } */
/* 	if (different) */
/* 	  tmp.push_back(points[i]); */
/*       } */
/*       points = tmp; */

/* #ifdef Augustdebug_cgal */
/*       std::cout << "After: " << points.size() << '\n'; */
/*       for (std::size_t i = 0; i < points.size(); ++i) */
/* 	std::cout << tools::matlabplot(points[i]); */
/*       std::cout << '\n'; */
/* #endif */

/*       if (points.size()<3) */
/*       { */
/* #ifdef Augustdebug_cgal */
/* 	std::cout << "too few points to form triangulation" << std::endl; */
/* #endif */
/* 	return triangulation; */
/*       } */


/*       // Do simple Graham scan */

/*       // Find left-most point (smallest x-coordinate) */
/*       std::size_t i_min = 0; */
/*       double x_min = points[0].x(); */
/*       for (std::size_t i = 1; i < points.size(); i++) */
/*       { */
/* 	const double x = points[i].x(); */
/* 	if (x < x_min) */
/* 	{ */
/* 	  x_min = x; */
/* 	  i_min = i; */
/* 	} */
/*       } */

/*       // Compute signed squared cos of angle with (0, 1) from i_min to all points */
/*       std::vector<std::pair<double, std::size_t>> order; */
/*       for (std::size_t i = 0; i < points.size(); i++) */
/*       { */
/* 	// Skip left-most point used as origin */
/* 	if (i == i_min) */
/* 	  continue; */

/* 	// Compute vector to point */
/* 	const Point v = points[i] - points[i_min]; */

/* 	// Compute square cos of angle */
/* 	const double cos2 = (v.y() < 0.0 ? -1.0 : 1.0)*v.y()*v.y() / v.squared_norm(); */

/* 	// Store for sorting */
/* 	order.push_back(std::make_pair(cos2, i)); */
/*       } */

/*       // Sort points based on angle */
/*       std::sort(order.begin(), order.end()); */

/*       // Triangulate polygon by connecting i_min with the ordered points */
/*       triangulation.reserve((points.size() - 2)*3*2); */
/*       const Point& p0 = points[i_min]; */
/*       for (std::size_t i = 0; i < points.size() - 2; i++) */
/*       { */
/* 	const Point& p1 = points[order[i].second]; */
/* 	const Point& p2 = points[order[i + 1].second]; */
/* 	triangulation.push_back(p0.x()); */
/* 	triangulation.push_back(p0.y()); */
/* 	triangulation.push_back(p1.x()); */
/* 	triangulation.push_back(p1.y()); */
/* 	triangulation.push_back(p2.x()); */
/* 	triangulation.push_back(p2.y()); */
/*       } */

/*       return triangulation; */
    }

    std::cout << "unexpected behavior in dolfin cgal tools, exiting"; exit(1);

    return std::vector<double>();
  }
}

namespace dolfin
{

  //---------------------------------------------------------------------------
  // Reference implementations using CGAL exact arithmetic
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
