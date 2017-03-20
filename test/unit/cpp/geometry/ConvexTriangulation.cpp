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
// Last changed: 2012-03-20
//
// Unit tests for the mesh library

#include <dolfin.h>
#include <gtest/gtest.h>

using namespace dolfin;

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

  // TOOD: Test that no triangles are degenerate and do not overlap
}
