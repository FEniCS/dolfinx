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
// Last changed: 2012-07-05
//
// This demo illustrates the built-in mesh types.

#include <dolfin.h>

using namespace dolfin;

int main()
{
  UnitInterval interval(10);
  info("Plotting a UnitInterval");
  plot(interval, "Unit interval");

  UnitSquare square_default(10, 10);
  info("Plotting a UnitSquare");
  plot(square_default, "Unit square");

  UnitSquare square_left(10, 10, "left");
  info("Plotting a UnitSquare");
  plot(square_left, "Unit square (left)");

  UnitSquare square_crossed(10, 10, "crossed");
  info("Plotting a UnitSquare");
  plot(square_crossed, "Unit square (crossed)");

  UnitSquare square_right_left(10, 10, "right/left");
  info("Plotting a UnitSquare");
  plot(square_right_left, "Unit square (right/left)");

  Rectangle rectangle_default(0.0, 0.0, 10.0, 4.0, 10, 10);
  info("Plotting a Rectangle");
  plot(rectangle_default, "Rectangle");

  Rectangle rectangle_right_left(-3.0, 2.0, 7.0, 6.0, 10, 10, "right/left");
  info("Plotting a Rectangle");
  plot(rectangle_right_left, "Rectangle (right/left)");

  UnitCircle circle_rotsumn(20, "right", "rotsumn");
  info("Plotting a UnitCircle");
  plot(circle_rotsumn, "Unit circle (rotsum)");

  //UnitCircle circle_sumn(20, "left", "sumn");
  //info("Plotting a UnitCircle");
  //plot(circle_sumn, "Unit circle (sumn)");

  UnitCircle circle_maxn(20, "right", "maxn");
  info("Plotting a UnitCircle");
  plot(circle_maxn, "Unit circle (maxn)");

  UnitCube cube(10, 10, 10);
  info("Plotting a UnitCube");
  plot(cube, "Unit cube");

  Box box(0.0, 0.0, 0.0, 10.0, 4.0, 2.0, 10, 10, 10);
  info("Plotting a Box");
  plot(box, "Box");

  UnitSphere sphere(10);
  info("Plotting a UnitSphere");
  plot(sphere, "Unit sphere");

  interactive();
  
  return 0;
}
