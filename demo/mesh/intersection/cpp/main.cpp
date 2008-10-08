// Copyright (C) 2008 Kristoffer Selim.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-10-08
// Last changed: 2008-10-08

#include <dolfin.h>
#include <math.h>

using namespace dolfin;

int main()
{
  // Create meshes
  UnitSquare omega1(20,20);  
  UnitCircle omega2(80);  
    
  // Move and scale circle
  for (VertexIterator vertex(omega2); !vertex.end(); ++vertex)
  {
    double* x = vertex->x();
    x[0] = 0.5*x[0] + 1.0;
    x[1] = 0.5*x[1] + 1.0;
  }

  // Iterate over angle
  const double dtheta = 0.10*2*DOLFIN_PI;
  for (double theta = 0; theta < 2*DOLFIN_PI; theta += dtheta)
  {
    // Compute intersection with boundary of square
    BoundaryMesh boundary(omega1);   
    Array<unsigned int> cells;
    omega2.intersection(boundary, cells, false);
    
    // Create mesh function to plot intersection
    MeshFunction<unsigned int> intersection(omega2, omega2.topology().dim());
    intersection = 0;
    for (unsigned int i = 0; i < cells.size(); i++)
      intersection.set(cells[i], 1);
    
    // Plot intersection
    plot(intersection);

    // Rotate circle around (0.5, 0.5)
    for (VertexIterator vertex(omega2); !vertex.end(); ++vertex)
    {
      double* x = vertex->x();
      const double x0 = x[0] - 0.5;
      const double x1 = x[1] - 0.5;
      x[0] = 0.5 + (cos(dtheta)*x0 - sin(dtheta)*x1);
      x[1] = 0.5 + (sin(dtheta)*x0 + cos(dtheta)*x1);
    }
  }
}
