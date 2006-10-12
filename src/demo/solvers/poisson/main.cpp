// Copyright (C) 2002-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2002-11-29
// Last changed: 2005-12-28
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
  real eval(const Point& p, unsigned int i)
  {
    return DOLFIN_PI*DOLFIN_PI*sin(DOLFIN_PI*p.x());
  }
};

// Boundary condition
class MyBC : public BoundaryCondition
{
  void eval(BoundaryValue& value, const Point& p, unsigned int i)
  {
    if ( (std::abs(p.x() - 0.0) < DOLFIN_EPS) || (std::abs(p.x() - 1.0) < DOLFIN_EPS ) )
      value = 0.0;
  }
};

int main()
{
  UnitSquare mesh(1, 1);
//   Mesh mesh("mesh.xml.gz");
//   mesh.refineUniformly(2);
  
  MyFunction f;
  MyBC bc;

  PoissonSolver::solve(mesh, f, bc);
  
  return 0;
}
