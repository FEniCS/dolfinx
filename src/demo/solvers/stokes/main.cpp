// Copyright (C) Anders Logg and Andy R. Terrel.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-09-20
// Last changed: 2005-09-20

#include <dolfin/StokesSolver.h>

using namespace dolfin;

// Right-hand side
class MyFunction : public Function
{
  real operator() (const Point& p, unsigned int i) const
  {
    return 0.0;
  }
};

// Boundary condition
class MyBC : public BoundaryCondition
{
  const BoundaryValue operator() (const Point& p, unsigned int i)
  {
    dolfin::cout << "Setting boundary condition at p = " << p << dolfin::endl;
    
    BoundaryValue value;
    if ( i == 0 && fabs(p.y - 1.0) )
      value = 1.0;
    //else if ( i == 0 || i == 1 )
    //  value = 0.0;
    
    return value;
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
