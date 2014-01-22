// Copyright (C) 2009 Kristian B. Oelgaard
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
// First added:  2009-09-29
// Last changed: 2012-12-12
//
// This demo illustrates the built-in mesh types.

#include <dolfin.h>

using namespace dolfin;

int main()
{
  if (dolfin::MPI::size(MPI_COMM_WORLD) == 1)
  {
    UnitIntervalMesh interval(10);
    info("Plotting a UnitIntervalMesh");
    plot(interval, "Unit interval");
  }

  UnitSquareMesh square_default(10, 10);
  info("Plotting a UnitSquareMesh");
  plot(square_default, "Unit square");

  UnitSquareMesh square_left(10, 10, "left");
  info("Plotting a UnitSquareMesh");
  plot(square_left, "Unit square (left)");

  UnitSquareMesh square_crossed(10, 10, "crossed");
  info("Plotting a UnitSquareMesh");
  plot(square_crossed, "Unit square (crossed)");

  UnitSquareMesh square_right_left(10, 10, "right/left");
  info("Plotting a UnitSquareMesh");
  plot(square_right_left, "Unit square (right/left)");

  RectangleMesh rectangle_default(0.0, 0.0, 10.0, 4.0, 10, 10);
  info("Plotting a RectangleMesh");
  plot(rectangle_default, "Rectangle");

  RectangleMesh rectangle_right_left(-3.0, 2.0, 7.0, 6.0, 10, 10, "right/left");
  info("Plotting a RectangleMesh");
  plot(rectangle_right_left, "Rectangle (right/left)");

  #ifdef HAS_CGAL
  CircleMesh circle_mesh(Point(0.0, 0.0), 1.0, 0.2);
  info("Plotting a CircleMesh");
  plot(circle_mesh, "Circle (unstructured)");

  std::vector<double> ellipse_dims(2);
  ellipse_dims[0] = 3.0; ellipse_dims[1] = 1.0;
  EllipseMesh ellipse_mesh(Point(0.0, 0.0), ellipse_dims, 0.2);
  info("Plotting an EllipseMesh");
  plot(ellipse_mesh, "Ellipse mesh (unstructured)");

  SphereMesh sphere_mesh(Point(0.0, 0.0, 0.0), 1.0, 0.2);
  info("Plotting a SphereMesh");
  plot(sphere_mesh, "Sphere mesh (unstructured)");

  std::vector<double> ellipsoid_dims(3);
  ellipsoid_dims[0] = 3.0; ellipsoid_dims[1] = 1.0; ellipsoid_dims[2] = 2.0;
  EllipsoidMesh ellipsoid_mesh(Point(0.0, 0.0, 0.0), ellipsoid_dims, 0.2);
  info("Plotting an EllipsoidMesh");
  plot(ellipsoid_mesh, "Ellipsoid mesh (unstructured)");
  #endif


  UnitCubeMesh cube(10, 10, 10);
  info("Plotting a UnitCubeMesh");
  plot(cube, "Unit cube");

  BoxMesh box(0.0, 0.0, 0.0, 10.0, 4.0, 2.0, 10, 10, 10);
  info("Plotting a BoxMesh");
  plot(box, "Box");

  interactive();

  return 0;
}
