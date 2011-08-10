// Copyright (C) 2008 Kristoffer Selim
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
// First added:  2010-01-14
// Last changed: 2011-08-10

#include <dolfin.h>
#include <math.h>

using namespace dolfin;

#ifdef HAS_CGAL

int main()
{
  // Create meshes (omega0 overlapped by omega1)
  UnitCircle omega0(20);
  UnitSquare omega1(20, 20);

  // Access mesh geometry
  MeshGeometry& geometry = omega0.geometry();

  // Move and scale circle
  for (VertexIterator vertex(omega0); !vertex.end(); ++vertex)
  {
    double* x = geometry.x(vertex->index());
    x[0] = 0.5*x[0] + 1.0;
    x[1] = 0.5*x[1] + 1.0;
  }

  // Iterate over angle
  const double dtheta = 0.1*DOLFIN_PI;
  for (double theta = 0; theta < 2*DOLFIN_PI; theta += dtheta)
  {
    // Compute intersection with boundary of square
    BoundaryMesh boundary(omega1);
    //typedef for std::set<unsigned int>
    std::set<dolfin::uint> cells;
    omega0.intersected_cells(boundary, cells);

    // Copy values to mesh function for plotting
    MeshFunction<unsigned int> intersection(omega0, omega0.topology().dim());
    intersection = 0;
    for (std::set<dolfin::uint>::const_iterator i = cells.begin(); i != cells.end(); i++)
      intersection[*i] = 1;

    // Plot intersection
    //plot(intersection);

    // Rotate circle around (0.5, 0.5)
    for (VertexIterator vertex(omega0); !vertex.end(); ++vertex)
    {
      double* x = geometry.x(vertex->index());
      const double xr = x[0] - 0.5;
      const double yr = x[1] - 0.5;
      x[0] = 0.5 + (cos(dtheta)*xr - sin(dtheta)*yr);
      x[1] = 0.5 + (sin(dtheta)*xr + cos(dtheta)*yr);
    }
  }
}

#else

int main()
{
  info("DOLFIN must be compiled with CGAL to run this demo.");
  return 0;
}

#endif
