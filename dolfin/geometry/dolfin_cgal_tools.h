#ifndef DOLFIN_CGAL_TOOLS_H
#define DOLFIN_CGAL_TOOLS_H

//#include "../dolfin.h" // for install
#include <dolfin.h> // for building test

#include <CGAL/Triangle_2.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/intersection_2.h>
#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/Polygon_set_2.h>


typedef CGAL::Exact_predicates_exact_constructions_kernel CGALKernel;
//typedef CGAL::Exact_predicates_inexact_constructions_kernel CGALKernel;
typedef CGALKernel::FT                            ExactNumber;
typedef CGAL::Point_2<CGALKernel>                 Point_2;
typedef CGAL::Triangle_2<CGALKernel>              Triangle_2;
typedef CGAL::Line_2<CGALKernel>                  Line_2;
typedef CGAL::Polygon_2<CGALKernel>               Polygon_2;
typedef Polygon_2::Vertex_const_iterator          Vertex_const_iterator;
typedef CGAL::Polygon_with_holes_2<CGALKernel>    Polygon_with_holes_2;
typedef Polygon_with_holes_2::Hole_const_iterator Hole_const_iterator;
typedef CGAL::Polygon_set_2<CGALKernel>           Polygon_set_2;

namespace cgaltools
{
  inline Point_2 convert(double a, double b)
  {
    return Point_2(a, b);
  }

  inline Point_2 convert(const dolfin::Point& p)
  {
    return Point_2(p[0], p[1]);
  }

  inline Line_2 convert(const dolfin::Point& a,
			const dolfin::Point& b)
  {
    return Line_2(convert(a), convert(b));
  }

  inline Triangle_2 convert(const dolfin::Point& a,
			    const dolfin::Point& b,
			    const dolfin::Point& c)
  {
    return Triangle_2(convert(a), convert(b), convert(c));
  }

}

#endif
