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
// First added:  2017-03-17
// Last changed: 2012-08-30
//
// Unit tests for convex triangulation

#include <dolfin.h>
#include <gtest/gtest.h>

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
      for (std::size_t i = 0; i < tri.size(); i++)
      {
	for (std::size_t j = i+1; j < tri.size(); j++)
	{
	  if ((tri[i]-tri[j]).norm() < DOLFIN_EPS)
	    return true;
	}
      }
    }
    return false;
  }

  // checks that every pair of cells share 0 or 1 face
  bool valid_topology(const std::vector<std::vector<Point>>& triangulation, std::size_t dim)
  {
    for (std::size_t i = 0; i < triangulation.size(); i++)
    {
      const auto& t1 = triangulation[i];
      for (std::size_t j = i+1; j < triangulation.size(); j++)
      {
	// Count number of shared vertices
	std::size_t shared_vertices = 0;
	for (std::size_t v1 = 0; v1 < dim+1; v1++)
	  for (std::size_t v2 = 0; v2 < dim+1; v2++)
	    if (v1 == v2)
	      shared_vertices++;

	if (shared_vertices != 0 && shared_vertices != dim)
	  return false;
      }
    }
    return true;
  }

  bool triangulation_selfintersects(const std::vector<std::vector<Point>>& triangulation, std::size_t dim)
  {
    for (std::size_t i = 0; i < triangulation.size(); i++)
    {
      const auto& t1 = triangulation[i];
      for (std::size_t j = i+1; j < triangulation.size(); j++)
      {
	// Count number of shared vertices
	std::size_t shared_vertices = 0;
	for (std::size_t v1 = 0; v1 < dim+1; v1++)
	  for (std::size_t v2 = 0; v2 < dim+1; v2++)
	    if (v1 == v2)
	      shared_vertices++;

	const auto& t2 = triangulation[j];


	// TODO: This should realli by improved as it currently only
	// simplices which are not neighbors...
	if ((dim == 3 &&
	     shared_vertices == 0 &&
	     CollisionPredicates::collides_tetrahedron_tetrahedron_3d(t1[0],
								      t1[1],
								      t1[2],
								      t1[3],
								      t2[0],
								      t2[1],
								      t2[2],
								      t2[3])) ||
	    (dim == 2 &&
	      shared_vertices == 0 &&
	     CollisionPredicates::collides_triangle_triangle_2d(t1[0],
								t1[1],
								t1[2],
								t2[0],
								t2[1],
								t2[2])))
	{
	  return true;
	}
      }
    }
    return false;
  }
}

//-----------------------------------------------------------------------------
TEST(ConvexTriangulationTest, testTrivialCase)
{
  std::vector<Point> input {
    Point(0,0,0),
    Point(0,0,1),
    Point(0,1,0),
    Point(1,0,0) };

  std::vector<std::vector<Point>> tri = ConvexTriangulation::triangulate_graham_scan_3d(input);

  ASSERT_EQ(tri.size(), 1);
}
//-----------------------------------------------------------------------------
TEST(ConvexTriangulationTest, testTrivialCase2)
{
  std::vector<Point> input {
    Point(0,0,0),
    Point(0,0,1),
    Point(0,1,0),
    Point(0,1,1),
    Point(1,0,0),
    Point(1,0,1),
    Point(1,1,0),
    Point(1,1,1) };

  std::vector<std::vector<Point>> tri = ConvexTriangulation::triangulate_graham_scan_3d(input);

  ASSERT_TRUE(pure_triangular(tri, 3));
  ASSERT_TRUE(!has_degenerate(tri, 3));
  ASSERT_TRUE(!triangulation_selfintersects(tri, 3));
}

TEST(ConvexTriangulationTest, testCoplanarPoints)
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
    Point(0.5, 0.5, 0)};

  std::vector<std::vector<Point>> tri = ConvexTriangulation::triangulate_graham_scan_3d(input);

  ASSERT_TRUE(pure_triangular(tri, 3));
  ASSERT_TRUE(!has_degenerate(tri, 3));
  ASSERT_TRUE(!triangulation_selfintersects(tri, 3));
}



TEST(ConvexTriangulationTest, testFailingCase)
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

  ASSERT_TRUE(pure_triangular(tri, 3));
  ASSERT_TRUE(!has_degenerate(tri, 3));
  ASSERT_FALSE(triangulation_selfintersects(tri, 3));
}

TEST(ConvexTriangulationTest, testFailingCase2)
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
    Point (0.7, 0.1, 0.38)
  };

  std::vector<std::vector<Point>> tri = ConvexTriangulation::triangulate_graham_scan_3d(input);

  ASSERT_TRUE(pure_triangular(tri, 3));
  ASSERT_TRUE(!has_degenerate(tri, 3));
  ASSERT_TRUE(!triangulation_selfintersects(tri, 3));
}

TEST(ConvexTriangulationTest, testFailingCase3)
{
  std::vector<Point> input {
    Point (0.495926, 0.512037, 0.144444),
    Point (0.376482, 0.519121, 0.284321),
    Point (0.386541, 0.599783, 0.0609262),
    Point (0.388086, 0.60059, 0.0607155),
    Point (0.7, 0.6, 0.5),
    Point (0.504965, 0.504965, 0.0447775),
    Point (0.833333, 0.833333, 0)
  };

  std::vector<std::vector<Point>> tri = ConvexTriangulation::triangulate_graham_scan_3d(input);

  ASSERT_TRUE(pure_triangular(tri, 3));
  ASSERT_TRUE(!has_degenerate(tri, 3));
  ASSERT_TRUE(!triangulation_selfintersects(tri, 3));
}

// Test all
int ConvecTriangulation_main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
