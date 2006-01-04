// Copyright (C) Anders Logg and Andy R. Terrel.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-09-20
// Last changed: 2005-12-30
//
// This demo solves the driven cavity test problem
// on the unit square with exact solution given by
//
//   u(x, y) = (-sin(pi*x)*cos(pi*y), cos(pi*x)*sin(pi*y))
//   p(x, y) = 0
//
// with corresponding right-hand side given by f(x, y) = 2*pi^2*u(x, y).

#include <dolfin/StokesSolver.h>
// FIXME: commented out to compile
// #include "L2Norm.h"

using namespace dolfin;

// Right-hand side
class MyFunction : public Function
{
  real eval(const Point& p, unsigned int i)
  {
    if ( i == 0 )
      return - 2.0 * DOLFIN_PI * DOLFIN_PI * sin(DOLFIN_PI*p.x) * cos(DOLFIN_PI*p.y);
    else
      return 2.0 * DOLFIN_PI * DOLFIN_PI * cos(DOLFIN_PI*p.x) * sin(DOLFIN_PI*p.y);
  }
};

// Boundary condition
class MyBC : public BoundaryCondition
{
  void eval(BoundaryValue& value, const Point& p, unsigned int i)
  {
    // Boundary condition for pressure
    if ( i == 2 )
    {
      value = 0.0;
      return;
    }

    // Boundary condition for velocity
    if ( std::abs(p.x - 0.0) < DOLFIN_EPS )
    {
      if ( i == 0 )
	value = 0.0;
      else
	value = sin(DOLFIN_PI*p.y);
    }
    else if ( std::abs(p.x - 1.0) < DOLFIN_EPS )
    {
      if ( i == 0 )
	value = 0.0;
      else
	value = - sin(DOLFIN_PI*p.y);
    }
    else if ( std::abs(p.y - 0.0) < DOLFIN_EPS )
    {
      if ( i == 0 )
	value = - sin(DOLFIN_PI*p.x);
      else
	value = 0.0;
    }
    else if ( std::abs(p.y - 1.0) < DOLFIN_EPS )
    {
      if ( i == 0 )
	value = sin(DOLFIN_PI*p.x);
      else
	value = 0.0;
    }
  }
};

int main()
{
  UnitSquare mesh(16, 16);
  MyFunction f;
  MyBC bc;
  
  StokesSolver::solve(mesh, f, bc);
  
  return 0;
}
