// Copyright (C) 2008 Kristoffer Selim.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-01-14
// Last changed: 2010-01-14

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
    uint_set cells;
    omega0.all_intersected_entities(boundary, cells);

    // Copy values to mesh function for plotting
    MeshFunction<unsigned int> intersection(omega0, omega0.topology().dim());
    intersection = 0;
    for (uint_set::const_iterator i = cells.begin(); i != cells.end(); i++)
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
