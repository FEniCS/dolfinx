// Copyright (C) 2017 Benjamin Kehlet
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
// Unit tests for convex triangulation

#include <dolfin/geometry/ConvexTriangulation.h>
#include <dolfin/geometry/CollisionPredicates.h>
#include <dolfin/geometry/predicates.h>
#include <catch.hpp>

using namespace dolfin;

namespace
{
  bool pure_triangular(const std::vector<std::vector<Point>>& triangulation, std::size_t dim)
  {
    for (const std::vector<Point>& tri : triangulation)
    {

      if (tri.size() != dim+1)
	return false;
    }

    return true;
  }

  bool has_degenerate(const std::vector<std::vector<Point>>& triangulation, std::size_t dim)
  {
    for (const std::vector<Point>& tri : triangulation)
    {
      if (orient3d(tri[0], tri[1], tri[2], tri[3]) == 0)
	return true;
    }
    return false;
  }

  bool triangulation_selfintersects(const std::vector<std::vector<Point>>& triangulation,
				    std::size_t dim)
  {
    for (std::size_t i = 0; i < triangulation.size(); i++)
    {
      const auto& t1 = triangulation[i];
      for (std::size_t j = i+1; j < triangulation.size(); j++)
      {
	const auto& t2 = triangulation[j];

	// Count number of shared vertices
	std::size_t shared_vertices = 0;
	std::set<std::size_t> t1_shared;
	std::set<std::size_t> t2_shared;
	for (std::size_t v1 = 0; v1 < dim+1; v1++)
	{
	  for (std::size_t v2 = 0; v2 < dim+1; v2++)
	  {
	    if (t1[v1] == t2[v2])
	    {
	      shared_vertices++;
	      t1_shared.insert(v1);
	      t2_shared.insert(v2);
	    }
	  }
	}

	if (dim == 3)
	{
	  if (shared_vertices == 0 &&
	      CollisionPredicates::collides_tetrahedron_tetrahedron_3d(t1[0], t1[1], t1[2], t1[3],
								       t2[0], t2[1], t2[2], t2[3]))
	  {
	    return true;
	  }
	  else if (shared_vertices > 0)
	  {

	    for (std::size_t a = 0; a < dim+1; a++)
	    {
	      // None of the non-shared vertices should collide with the other tet
	      if (t1_shared.count(a) == 0 &&
		  CollisionPredicates::collides_tetrahedron_point_3d(t2[0], t2[1], t2[2], t2[3],
								     t1[a]))
	      {
		return true;
	      }

	      if (t2_shared.count(a) == 0 &&
		  CollisionPredicates::collides_tetrahedron_point_3d(t1[0], t1[1], t1[2], t1[3],
								     t2[a]))
	      {
		return true;
	      }
	    }
	  }
	}
	else if(dim == 2)
	{
	  if (shared_vertices == 0 &&
	      CollisionPredicates::collides_triangle_triangle_2d(t1[0],
								 t1[1],
								 t1[2],
								 t2[0],
								 t2[1],
								 t2[2]))
	  {
	    return true;
	  }
	}
      }
    }
    return false;
  }
  //-----------------------------------------------------------------------------
  double triangulation_volume(const std::vector<std::vector<dolfin::Point>>& triangulation)
  {
    double vol = 0;
    for (const std::vector<dolfin::Point>& tri : triangulation)
    {
      const Point& x0 = tri[0];
      const Point& x1 = tri[1];
      const Point& x2 = tri[2];
      const Point& x3 = tri[3];
      // Formula for volume from http://mathworld.wolfram.com
      const double v = (x0[0]*(x1[1]*x2[2] + x3[1]*x1[2] + x2[1]*x3[2]
			     - x2[1]*x1[2] - x1[1]*x3[2] - x3[1]*x2[2])
		      - x1[0]*(x0[1]*x2[2] + x3[1]*x0[2] + x2[1]*x3[2]
			     - x2[1]*x0[2] - x0[1]*x3[2] - x3[1]*x2[2])
                      + x2[0]*(x0[1]*x1[2] + x3[1]*x0[2] + x1[1]*x3[2]
			     - x1[1]*x0[2] - x0[1]*x3[2] - x3[1]*x1[2])
		      - x3[0]*(x0[1]*x1[2] + x1[1]*x2[2] + x2[1]*x0[2]
			     - x1[1]*x0[2] - x2[1]*x1[2] - x0[1]*x2[2]));
      vol += std::abs(v);
    }

    return vol/6;
  }
}

//-----------------------------------------------------------------------------
TEST_CASE("Convex triangulation test")
{
  SECTION("test trivial case]")
  {
    std::vector<Point> input = {{Point(0,0,0),
                                 Point(0,0,1),
                                 Point(0,1,0),
                                 Point(1,0,0)}};

    std::vector<std::vector<Point>> tri = ConvexTriangulation::triangulate_graham_scan_3d(input);
    std::size_t expected_size = 1;

    CHECK(tri.size() == expected_size);
    CHECK(triangulation_volume(tri) == Approx(1.0/6.0));
  }

  SECTION("test trivial case 2")
  {
    std::vector<Point> input = {{Point(0,0,0),
                                 Point(0,0,1),
                                 Point(0,1,0),
                                 Point(0,1,1),
                                 Point(1,0,0),
                                 Point(1,0,1),
                                 Point(1,1,0),
                                 Point(1,1,1)}};

    std::vector<std::vector<Point>> tri = ConvexTriangulation::triangulate_graham_scan_3d(input);

    CHECK(pure_triangular(tri, 3));
    CHECK_FALSE(has_degenerate(tri, 3));
    CHECK_FALSE(triangulation_selfintersects(tri, 3));
    CHECK(triangulation_volume(tri) == Approx(1.0));
  }

  SECTION("test coplanar points")
  {
    std::vector<Point> input {
      Point(0,   0,   0),
        Point(0,   0,   1),
        Point(0,   1,   0),
        Point(0,   1,   1),
        Point(1,   0,   0),
        Point(1,   0,   1),
        Point(1,   1,   0),
        Point(1,   1,   1),
        Point(0.1, 0.1, 0)};

    std::vector<std::vector<Point>> tri = ConvexTriangulation::triangulate_graham_scan_3d(input);

    CHECK(pure_triangular(tri, 3));
    CHECK_FALSE(has_degenerate(tri, 3));
    CHECK_FALSE(triangulation_selfintersects(tri, 3));
    CHECK(triangulation_volume(tri) == Approx(1.0));
  }

  SECTION("test coplanar colinear points]")
  {
    std::vector<Point> input {
      Point(0, 0,   0),
        Point(0, 0,   1),
        Point(0, 1,   0),
        Point(0, 1,   1),
        Point(1, 0,   0),
        Point(1, 0,   1),
        Point(1, 1,   0),
        Point(1, 1,   1),
        Point(0, 0.1, 0)};

    std::vector<std::vector<Point>> tri = ConvexTriangulation::triangulate_graham_scan_3d(input);

    CHECK(pure_triangular(tri, 3));
    CHECK_FALSE(has_degenerate(tri, 3));
    CHECK_FALSE(triangulation_selfintersects(tri, 3));
    CHECK(triangulation_volume(tri) == Approx(1.0));
  }

  SECTION("test failing case")
  {
    std::vector<Point> input {
      Point(0.7, 0.6, 0.1),
        Point(0.7, 0.6, 0.5),
        Point(0.1, 0.1, 0.1),
        Point(0.8333333333333333, 0.8333333333333333, 0),
        Point(0.1, 0.15, 0.1),
        Point(0.1, 0.45, 0.1),
        Point(0.16, 0.15, 0.1),
        Point(0.61, 0.525, 0.1),
        Point(0.46, 0.6, 0.100000000000000006) };

    std::vector<std::vector<Point>> tri = ConvexTriangulation::triangulate_graham_scan_3d(input);

    CHECK(pure_triangular(tri, 3));
    CHECK_FALSE(has_degenerate(tri, 3));
    CHECK_FALSE(triangulation_selfintersects(tri, 3));
  }

  SECTION("test failing case 2")
  {
    std::vector<Point> input {
      Point(0.7, 0.6, 0.5),
        Point(0.7, 0.1, 0.1),
        Point(0.8, 0, 0),
        Point (0.1, 0.1, 0.1),
        Point(0.16, 0.1, 0.1),
        Point(0.592, 0.1, 0.1),
        Point (0.16, 0.1, 0.14),
        Point (0.52, 0.1, 0.38),
        Point (0.7, 0.1, 0.38)};

    std::vector<std::vector<Point>> tri = ConvexTriangulation::triangulate_graham_scan_3d(input);

    CHECK(pure_triangular(tri, 3));
    CHECK_FALSE(has_degenerate(tri, 3));
    CHECK_FALSE(triangulation_selfintersects(tri, 3));
  }

  SECTION("test failing case 3")
  {
    std::vector<Point> input {
      Point (0.495926, 0.512037, 0.144444),
        Point (0.376482, 0.519121, 0.284321),
        Point (0.386541, 0.599783, 0.0609262),
        Point (0.388086, 0.60059, 0.0607155),
        Point (0.7, 0.6, 0.5),
        Point (0.504965, 0.504965, 0.0447775),
        Point (0.833333, 0.833333, 0)};

    std::vector<std::vector<Point>> tri = ConvexTriangulation::triangulate_graham_scan_3d(input);

    CHECK(pure_triangular(tri, 3));
    CHECK_FALSE(has_degenerate(tri, 3));
    CHECK_FALSE(triangulation_selfintersects(tri, 3));
  }
}
//-----------------------------------------------------------------------------
