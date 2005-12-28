// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2005.
//
// First added:  ?
// Last changed: 2005-12-28

#include <dolfin.h>
  
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

  set("method", "dg");
  set("order", 0);
  set("solver", "newton");
  set("tolerance", 1e-3);

  set("fixed time step", false);
  set("initial time step", 1e-1);
  set("maximum time step", 1e6);

  set("save solution", true);
  set("ode solution file name", "primal.py");
  set("number of samples", 400);

  real T = 10.0;

  HeatSolver::solve(mesh, f, bc, T);

  return 0;
}
