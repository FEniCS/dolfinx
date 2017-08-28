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
// Last changed: 2012-08-28
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

      if (tri.size() == dim+1)
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

  bool triangulation_selfintersects(const std::vector<std::vector<Point>>& triangulation, std::size_t dim)
  {
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
  ASSERT_TRUE(!triangulation_selfintersects(tri, 3));
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
