// Copyright (C) 2012 Garth N. Wells
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
// First added:  2012-02-02
// Last changed:
//
// This demo creates meshes from the triangulation of a collection of
// random point.

#include <dolfin.h>

using namespace dolfin;

#ifdef HAS_CGAL

int main()
{
  // Generate collection of random points
  const uint num_points = 2000;
  std::vector<Point> random_points;
  for (uint i = 0; i < num_points; ++i)
    random_points.push_back(Point(dolfin::rand(), dolfin::rand(), dolfin::rand()));

  // Create empty Mesh
  Mesh mesh;

  // Triangulate points in 2D and plot mesh
  Triangulate::triangulate(mesh, random_points, 2);
  plot(mesh);

  // Triangulate points in 3D and plot mesh
  Triangulate::triangulate(mesh, random_points, 3);
  plot(mesh);
}

#else

int main()
{
  info("DOLFIN must be compiled with CGAL to run this demo.");
}

#endif
