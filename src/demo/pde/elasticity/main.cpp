// Copyright (C) 2006 Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-02-07
// Last changed: 2006-02-07
//
// This demo program solves the equations of static
// linear elasticity for a gear clamped at two of its
// ends and twisted 30 degrees.

#include <dolfin.h>
#include "Elasticity.h"

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
    // Width of clamp
    real w = 0.1;

    // Center of rotation
    real y0 = 0.5;
    real z0 = 0.219;

    // Angle of rotation (30 degrees)
    real theta = 0.5236;

    // New coordinates
    real y = y0 + (p.y - y0)*cos(theta) - (p.z - z0)*sin(theta);
    real z = z0 + (p.y - y0)*sin(theta) + (p.z - z0)*cos(theta);
    
    // Clamp at left end
    if ( p.x < w )
      value = 0.0;

    // Clamp at right end
    if ( p.x > (1.0 - w) )
    {
      if ( i == 1 )
	value = y - p.y;
      else if ( i == 2 )
	value = z - p.z;
    }
  }
};

int main()
{
  // Set up problem
  Mesh mesh("gear.xml.gz");
  MyFunction f;
  MyBC bc;
  Elasticity::BilinearForm a;
  Elasticity::LinearForm L(f);

  // Assemble linear system
  Matrix A;
  Vector x, b;
  FEM::assemble(a, L, A, b, mesh, bc);
  
  // Solve the linear system
  LU solver;
  solver.solve(A, x, b);
  
  // Save function to file
  Function u(x, mesh, a.trial());
  File file("elasticity.pvd");
  file << u;

  return 0;
}
