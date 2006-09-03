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
    return 10.0 * p.x*sin(p.y);
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

  set("ODE method", "dg");
  set("ODE order", 0);
  set("ODE nonlinear solver", "newton");
  set("ODE tolerance", 1e-3);

  set("ODE fixed time step", false);
  set("ODE initial time step", 1e-3);
  set("ODE maximum time step", 1e+0);

  set("ODE save solution", true);
  set("ODE solution file name", "primal.py");
  set("ODE number of samples", 400);

  real T = 10.0;

  HeatSolver::solve(mesh, f, bc, T);

  return 0;
}
