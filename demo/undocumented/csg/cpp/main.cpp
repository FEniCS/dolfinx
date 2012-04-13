// Copyright (C) 2012 Anders Logg
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
// First added:  2012-04-13
// Last changed: 2012-04-13

#include <dolfin.h>

using namespace dolfin;

int main()
{

  // Test some 2D geometries
  csg::Rectangle r(0, 0, 1, 1);
  csg::Circle c(0.5, 0.5, 1);
  boost::shared_ptr<CSGGeometry> g2d = c*(r + c)*(c + c) + r;

  // Test printing
  info("\nCompact output of 2D geometry:");
  info(*g2d);
  info("");
  info("\nVerbose output of 2D geometry:");
  info(*g2d, true);

  // Test some 3D geometries
  csg::Box b(0, 0, 0, 1, 1, 1);
  csg::Sphere s(0, 0, 0, 1);
  boost::shared_ptr<const CSGGeometry> g3d = (b + s)*b*(s + s) + b;

  // Test printing
  info("\nCompact output of 3D geometry:");
  info(*g3d);
  info("\nVerbose output of 3D geometry:");
  info(*g3d, true);

  return 0;
}
