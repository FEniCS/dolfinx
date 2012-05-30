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
// Modified by Johannes Ring, 2012
//
// First added:  2012-04-13
// Last changed: 2012-05-30

#include <dolfin.h>

using namespace dolfin;

#ifdef HAS_CGAL

int main()
{

  // Define 2D geometry
  csg::Rectangle r(0.5, 0.5, 1.5, 1.5);
  csg::Circle c(1, 1, 1);
  boost::shared_ptr<CSGGeometry> g2d = c - r;

  // Test printing
  info("\nCompact output of 2D geometry:");
  info(*g2d);
  info("");
  info("\nVerbose output of 2D geometry:");
  info(*g2d, true);

  // Generate mesh
  Mesh mesh2d(g2d);

  // Plot meshes
  plot(mesh2d, "2D mesh");

  return 0;
}

#else

int main()
{
  info("DOLFIN must be compiled with CGAL to run this demo.");
  return 0;
}

#endif
