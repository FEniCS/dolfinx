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
// Modified by Benjamin Kehlet, 2012
//
// First added:  2012-04-13
// Last changed: 2012-04-28

#include <dolfin.h>

using namespace dolfin;

int main()
{

  // // Define 2D geometry
  // csg::Rectangle r(0, 0, 1, 1);
  // csg::Circle c(0.5, 0.5, 1);
  // boost::shared_ptr<CSGGeometry> g2d = c*(r + c)*(c + c) + r;

  // // Test printing
  // info("\nCompact output of 2D geometry:");
  // info(*g2d);
  // info("");
  // info("\nVerbose output of 2D geometry:");
  // info(*g2d, true);

  // // Generate mesh
  // Mesh mesh2d(g2d);

  // Define 3D geometry
  csg::Box a(0, 0, 0, 1.0, 1.0, 1.0);
  csg::Box b(.25, .25, .25, .75, .75, 1.5);
  <const CSGGeometry g3d = a+b;

  // Test printing
  info("\nCompact output of 3D geometry:");
  info(*g3d);
  info("\nVerbose output of 3D geometry:");
  info(*g3d, true);

  // Generate mesh
  Mesh mesh3d(g3d);

  // Plot meshes
  //plot(mesh2d, "2D mesh");
  plot(mesh3d, "3D mesh");

  return 0;
}
