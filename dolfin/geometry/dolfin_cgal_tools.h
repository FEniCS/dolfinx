#ifndef DOLFIN_CGAL_TOOLS_H
#define DOLFIN_CGAL_TOOLS_H

#include <CGAL/Triangle_2.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/intersection_2.h>
#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/Polygon_set_2.h>

#include <dolfin/geometry/dolfin_simplex_tools.h>
#include <dolfin/geometry/IntersectionTriangulation.h>


#define Augustcgal
#define Augustdebug_cgal

typedef CGAL::Exact_predicates_exact_constructions_kernel CGALKernel;
//typedef CGAL::Exact_predicates_inexact_constructions_kernel CGALKernel;
typedef CGALKernel::FT                            ExactNumber;
typedef CGAL::Point_2<CGALKernel>                 Point_2;
typedef CGAL::Triangle_2<CGALKernel>              Triangle_2;
typedef CGAL::Segment_2<CGALKernel>               Segment_2;

namespace cgaltools
{
  inline Point_2 convert(double a, double b)
  {
    return Point_2(a, b);
  }

  inline Point_2 convert(const dolfin::Point& p)
  {
    //std::cout << "point convert " << p[0]<<' '<<p[1]<<std::endl;
    return Point_2(p[0], p[1]);
  }

  inline Segment_2 convert(const dolfin::Point& a,
			   const dolfin::Point& b)
  {
    //std::cout << "segment convert " << std::endl;
    return Segment_2(convert(a), convert(b));
  }

  inline Triangle_2 convert(const dolfin::Point& a,
			    const dolfin::Point& b,
			    const dolfin::Point& c)
  {
    //std::cout << "triangle convert " << std::endl;
    return Triangle_2(convert(a), convert(b), convert(c));
  }

  inline std::vector<dolfin::Point> convert(const Segment_2& t)
  {
    std::vector<dolfin::Point> p(2);
    for (std::size_t i = 0; i < 2; ++i)
      for (std::size_t j = 0; j < 2; ++j)
	p[i][j] = CGAL::to_double(t[i][j]);
    return p;
  }

  inline std::vector<dolfin::Point> convert(const Triangle_2& t)
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
    Segment_2 s(convert(a), convert(b));
    return s.is_degenerate();
  }

  inline bool is_degenerate(const dolfin::Point& a,
			    const dolfin::Point& b,
			    const dolfin::Point& c)
  {
    Triangle_2 t(convert(a), convert(b), convert(c));
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

#endif
