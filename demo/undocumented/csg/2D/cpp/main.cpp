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
// Modified by Joachim B Haga, 2012
//
// First added:  2012-04-13
// Last changed: 2012-11-12

#include <dolfin.h>

using namespace dolfin;

#ifdef HAS_CGAL

int main()
{
  // Define 2D geometry
  // Rectangle r(0.5, 0.5, 1.5, 1.5);
  // Circle c(1, 1, 1);
  // boost::shared_ptr<CSGGeometry> g2d = c - r;

  // Define 2D geometry
  Rectangle r1(0., 0., 5., 5.);
  Rectangle r2 (2., 1.25, 3., 1.75);
  Circle c1(1, 4, .25);
  Circle c2(4, 4, .25);
  boost::shared_ptr<CSGGeometry> domain =  r1 - r2 - c1 - c2;

  Rectangle s1(1., 1., 4., 3.);
  domain->set_subdomain(1, s1);

  Rectangle s2(2., 2., 3., 4.);
  domain->set_subdomain(2, s2);


  // Test printing
  info("\nCompact output of 2D geometry:");
  info(*domain);
  info("");
  info("\nVerbose output of 2D geometry:");
  info(*domain, true);

  // Plot geometry
  plot(domain, "2D Geometry (boundary)");

  // Generate and plot mesh
  Mesh mesh2d(domain, 100);
  plot(mesh2d, "2D mesh");
  plot(mesh2d.domains().cell_domains(), "Subdomains");


  interactive();
  return 0;
}

#else

int main()
{
  info("DOLFIN must be compiled with CGAL to run this demo.");
  return 0;
}

#endif
