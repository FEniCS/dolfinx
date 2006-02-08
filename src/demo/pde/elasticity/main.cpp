// Copyright (C) 2006 Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-02-07
// Last changed: 2006-02-07
//
// This demo program solves the equations of static
// linear elasticity
//
//     - div sigma(u) = f
//
// on the unit square with source f given by
//
//     f(x, y) = x * sin(y)
//
// and boundary conditions given by
//
//     u(x, y)     = 0  for x = 0
//     du/dn(x, y) = 0  otherwise

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

    // Angle of rotation
    real theta = 0.1;

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
	value = y;
      else if ( i == 2 )
	value = z;
    }
  }
};

int main()
{
  // Set up problem
  //UnitCube mesh(16, 16, 16);
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
  GMRES solver;
  solver.set("monitor convergence", true);
  solver.disp();
  solver.solve(A, x, b);
  
  // Save function to file
  Function u(x, mesh, a.trial());
  File file("elasticity.pvd");
  file << u;
  
  return 0;
}
