// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-02-09
// Last changed: 2006-02-09

#include <dolfin.h>
#include "Stokes.h"

using namespace dolfin;

// Right-hand side
class MyFunction : public Function
{
  real eval(const Point& p, unsigned int i)
  {
    return 0.0;
  }
};

// Boundary condition
class MyBC : public BoundaryCondition
{
  void eval(BoundaryValue& value, const Point& p, unsigned int i)
  {
    // Pressure boundary condition, zero pressure at one point
    if ( i == 2 && p.x < DOLFIN_EPS && p.y < DOLFIN_EPS )
    {
      value = 0.0;
      return;
    }

    // Velocity boundary condition at inflow
    if ( p.x > (1.0 - DOLFIN_EPS) )
    {
      if ( i == 0 )
	value = -1.0;
      else
	value = 0.0;
      return;
    }
    
    // Velocity boundary condition at remaining boundary (excluding outflow)
    if ( p.x > DOLFIN_EPS )
      value = 0.0;
  }
};

int main()
{
  // Set up problem
  Mesh mesh("dolfin-2.xml.gz");
  MyFunction f;
  MyBC bc;
  Stokes::BilinearForm a;
  Stokes::LinearForm L(f);

  // Assemble linear system
  Matrix A;
  Vector x, b;
  FEM::assemble(a, L, A, b, mesh, bc);

  // Solve the linear system
  GMRES solver;
  solver.solve(A, x, b);

  // Pick the two sub functions of the solution
  Function w(x, mesh, a->trial());
  Function u = w[0];
  Function p = w[1];

  // Save the solutions to file
  File ufile("velocity.pdv");
  File pfile("pressure.pvd");
  ufile << u;
  pfile << p;
}
