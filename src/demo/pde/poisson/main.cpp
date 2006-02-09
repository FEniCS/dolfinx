// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005
// Last changed: 2005-12-28
//
// This demo program solves Poisson's equation
//
//     - div grad u(x, y) = f(x, y)
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
#include "Poisson.h"
  
using namespace dolfin;

// Right-hand side
class MyFunction : public Function
{
  real eval(const Point& p, unsigned int i)
  {
    return p.x*sin(p.y);
  }
};

// Boundary condition
class MyBC : public BoundaryCondition
{
  void eval(BoundaryValue& value, const Point& p, unsigned int i)
  {
    if ( std::abs(p.x - 1.0) < DOLFIN_EPS )
      value = 0.0;
  }
};

int main()
{
  // Set up problem
  UnitSquare mesh(16, 16);
  MyFunction f;
  MyBC bc;
  Poisson::BilinearForm a;
  Poisson::LinearForm L(f);

  // Assemble linear system
  Matrix A;
  Vector x, b;
  FEM::assemble(a, L, A, b, mesh, bc);
  
  // Solve the linear system
  GMRES solver;
  solver.solve(A, x, b);
  
  // Save function to file
  Function u(x, mesh, a.trial());
  File file("poisson.pvd");
  file << u;
  
  return 0;
}
