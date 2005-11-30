// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2005.
//
// First added:  ?
// Last changed: 2005-11-29

#include <dolfin.h>
  
using namespace dolfin;

// Right-hand side
class MyFunction : public Function
{
  real operator() (const Point& p, unsigned int i)
  {
    return p.x*sin(p.y);
  }
};

// Boundary condition
class MyBC : public BoundaryCondition
{
  const BoundaryValue operator() (const Point& p)
  {
    BoundaryValue value;
    if ( std::abs(p.x - 1.0) < DOLFIN_EPS )
      value = 0.0;
    return value;
  }
};

int main()
{
  // Set up problem
  UnitSquare mesh(16, 16);
  MyFunction f;
  MyBC bc;

  dolfin_set("method", "dg");
  dolfin_set("order", 0);
  dolfin_set("solver", "newton");
  dolfin_set("tolerance", 1e-3);

  dolfin_set("fixed time step", false);
  dolfin_set("initial time step", 1e-1);
  dolfin_set("maximum time step", 1e6);

  dolfin_set("save solution", true);
  dolfin_set("file name", "primal.py");
  dolfin_set("number of samples", 400);

  real T = 10.0;

  HeatSolver::solve(mesh, f, bc, T);

  return 0;
}
