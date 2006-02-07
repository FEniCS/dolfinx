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
    if ( p.x > (1.0 - DOLFIN_EPS) && i == 2 )
      return -10.0;
    else
      return 0.0;
  }
};

// Boundary condition
class MyBC : public BoundaryCondition
{
  void eval(BoundaryValue& value, const Point& p, unsigned int i)
  {
    if ( p.x < DOLFIN_EPS )
      value = 0.0;
  }
};

int main()
{
  // Set up problem
  UnitCube mesh(16, 16, 16);
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
  solver.solve(A, x, b);
  
  // Save function to file
  Function u(x, mesh, a.trial());
  File file("elasticity.pvd");
  file << u;
  
  return 0;
}
