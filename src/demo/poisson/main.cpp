// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// This simple program solves Poisson's equation
//
//     - div grad u(x, y) = f(x, y)
//
// on the unit square with source f given by
//
//     f(x, y) = pi^2 * sin(pi*x)
//
// and boundary conditions given by
//
//     u(x, y)     = 0  for x = 0 or x = 1,
//     du/dn(x, y) = 0  for y = 0 or y = 1.
//
// The exact solution is given by u(x, y) = sin(pi*x).

#include <dolfin.h>
#include "Poisson.h"

using namespace dolfin;

// Right-hand side
class MyFunction : public Function
{
  real operator() (const Point& p) const
  {
    return DOLFIN_PI*DOLFIN_PI*sin(DOLFIN_PI*p.x);
  }
};

// Boundary condition
class MyBC : public BoundaryCondition
{
  const BoundaryValue operator() (const Point& p)
  {
    BoundaryValue value;
    if ( (std::abs(p.x - 0.0) < DOLFIN_EPS) || (std::abs(p.x - 1.0) < DOLFIN_EPS ) )
      value.set(0.0);
    
    return value;
  }
};

int main()
{
  // Set up problem
  Mesh mesh("mesh.xml.gz");
  MyFunction f;
  MyBC bc;
  Poisson::BilinearForm a;
  Poisson::LinearForm L(f);
  
  // Discretize equation
  Matrix A;
  Vector x, b;
  FEM::assemble(a, L, A, b, mesh, bc);

  // Solve the linear system
  GMRES solver;
  solver.solve(A, x, b);

  // Save function to file
  Function u(x, mesh, a.trial());
  File file("poisson.m");
  file << u;

  return 0;
}
