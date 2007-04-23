// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-02-09
// Last changed: 2006-10-18

#include <dolfin.h>
#include "Stokes.h"

using namespace dolfin;

int main()
{
  // Boundary condition
  class MyBC : public BoundaryCondition
  {
    void eval(BoundaryValue& value, const Point& p, unsigned int i)
    {
      // Pressure boundary condition, zero pressure at one point
      if ( i == 2 )
      {
        if ( p.x() < DOLFIN_EPS && p.y() < DOLFIN_EPS )
	        value = 0.0;
        return;
      }
      
      // Velocity boundary condition at inflow
      if ( p.x() > (1.0 - DOLFIN_EPS) )
      {
        if ( i == 0 )
          value = -1.0;
        else
          value = 0.0;
        return;
      }
      
      // Velocity boundary condition at remaining boundary (excluding outflow)
      if ( p.x() > DOLFIN_EPS )
        value = 0.0;
    }
  };

  // Set up problem
  Mesh mesh("../../../../../data/meshes/dolfin-2.xml.gz");
  Function f = 0.0;
  MeshSize h;
  MyBC bc;
  Stokes::BilinearForm a(h);
  Stokes::LinearForm L(f, h);
  PDE pde(a, L, mesh, bc);

  // Compute solution
  Function U;
  Function P;
  pde.solve(U, P);

  // Save solution to file
  File ufile("velocity.pvd");
  File pfile("pressure.pvd");
  ufile << U;
  pfile << P;
}
