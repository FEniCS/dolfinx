// Copyright (C) Anders Logg and Andy R. Terrel.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-09-20
// Last changed: 2005-12-28
//
// This demo solves the driven cavity test problem
// on the unit square.

#include <dolfin/StokesSolver.h>

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
    if ( i == 0 && fabs(p.y - 1.0) < DOLFIN_EPS )
      value = 1.0;
    else if ( i == 0 || i == 1 )
      value = 0.0;
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
