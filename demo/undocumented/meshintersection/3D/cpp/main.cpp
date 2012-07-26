// Copyright (C) 2012 Benjamin Kehlet
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
// First added:  2012-07-19
// Last changed: 2012-07-19

#include <dolfin.h>

using namespace dolfin;

#ifdef HAS_CGAL

int main()
{
  UnitSphere sphere(20);
  UnitCube cube(20, 20, 20);

  // Access mesh geometry
  MeshGeometry& geometry = sphere.geometry();

  // Start center and propagtion speed for the sphere.
  const double dt = 0.1;
  double t = -0.61;

  // Scale and move the circle.
  for (VertexIterator vertex(sphere); !vertex.end(); ++vertex)
  {
    double* x = geometry.x(vertex->index());
    x[0] = x[0]*0.7 + t;
    x[1] = x[1]*0.7 + t;
    x[2] = x[2]*0.7 + t;
  }

  boost::shared_ptr<dolfin::MeshFunction<unsigned int> >intersection(new dolfin::MeshFunction<unsigned int>(cube, cube.topology().dim()));

  VTKPlotter p(intersection);
  p.parameters["scalarbar"] = false;

  bool first = true;
  while (t < 1.4) 
  {
    // Compute intersection with boundary of square
    BoundaryMesh boundary(sphere);
    std::set<dolfin::uint> cells;
    cube.intersected_cells(boundary, cells);

    // Copy values to mesh function for plotting
    *intersection = 0;
    for (std::set<dolfin::uint>::const_iterator i = cells.begin(); i != cells.end(); i++)
      (*intersection)[*i] = 1;

    // Plot intersection
    p.plot();
    if (first)
    {
      p.elevate(-50);
      p.azimuth(40);
      first = false;
    }

    // Propagate sphere along the line t(1,1,1).
    for (VertexIterator vertex(sphere); !vertex.end(); ++vertex)
    {
      double* x = geometry.x(vertex->index());
      x[0] += dt;
      x[1] += dt;
      x[2] += dt;
    }

    t += dt;
  }

  // Hold plot
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
