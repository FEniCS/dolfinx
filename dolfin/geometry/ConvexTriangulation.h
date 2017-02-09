// Copyright (C) 2016 Anders Logg, August Johansson and Benjamin Kehlet
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
// First added:  2016-06-01
// Last changed: 2017-02-06

#ifndef __CONVEX_TRIANGULATION
#define __CONVEX_TRIANGULATION

#include "Point.h"
#include "CGALExactArithmetic.h"
#include <vector>

namespace dolfin
{

  class ConvexTriangulation
  {
  public:

    // Tdim independent wrapper
    static std::vector<std::vector<Point>>
    triangulate(std::vector<Point> p,
                std::size_t gdim,
                std::size_t tdim);

    // Triangulate using the Graham scan
    static std::vector<std::vector<Point>>
    triangulate_graham_scan_2d(std::vector<Point> pm)
    {
      // return CHECK_CGAL(_triangulate_graham_scan_2d(pm),
      // 			cgal_triangulate_2d(pm));
      return _triangulate_graham_scan_2d(pm);
    }

    static std::vector<std::vector<Point>>
    triangulate_graham_scan_3d(std::vector<Point> pm)
    {
      // return CHECK_CGAL(_triangulate_graham_scan_3d(pm),
      // 			cgal_triangulate_3d(pm));
      return _triangulate_graham_scan_3d(pm);
    }

  private:

    // Implementation declarations
    static std::vector<std::vector<Point>>
    _triangulate_graham_scan_2d(std::vector<Point> pm);

    static std::vector<std::vector<Point>>
    _triangulate_graham_scan_3d(std::vector<Point> pm);

    // Triangulate using Bowyer-Watson (Delaunay)
    static std::vector<std::vector<Point>>
    _triangulate_bowyer_watson(std::vector<Point> p,
			       std::size_t gdim);



    // Help class for Bowyer Watson
    struct Edge
    {
      Edge(const Point p0, const Point p1) : p0(p0), p1(p1) {}
      Edge(const Edge& e) : p0(e.p0), p1(e.p1) {}
      bool operator==(const Edge& e) const;

      Point p0;
      Point p1;
    };

    // Help class for Bowyer Watson
    struct Triangle
    {
      Triangle(const Point p0,
	       const Point p1,
	       const Point p2)
	: p0(p0), p1(p1), p2(p2),
	  e0(p0, p1), e1(p1, p2), e2(p2, p0)
      {}
      bool contains_vertex(const Point v) const;
      bool circumcircle_contains(const Point v) const;
      bool operator==(const Triangle& t) const;

      Point p0;
      Point p1;
      Point p2;
      Edge e0;
      Edge e1;
      Edge e2;
    };

    // Utility functions
    static std::vector<Point> unique_points(const std::vector<Point>& points,
					    double tol);
    static Point cross_product(Point a,
			       Point b,
			       Point c);
  };

} // end namespace dolfin
#endif
