// Copyright (C) 2002-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2002-11-29
// Last changed: 2005-09-20
//
// A simple test program for the Poisson solver, solving
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

#include <dolfin/PoissonSolver.h>

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
    if ( (fabs(p.x - 0.0) < DOLFIN_EPS) || (fabs(p.x - 1.0) < DOLFIN_EPS ) )
      value.set(0.0);
    
    return value;
  }
};

int main()
{
  Mesh mesh("mesh.xml.gz");
  MyFunction f;
  MyBC bc;
  
  PoissonSolver::solve(mesh, f, bc);
  
  return 0;
}
