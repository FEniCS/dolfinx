// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005
// Last changed: 2006-03-24
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
//     u(x, y)     = 0  for x = 1
//     du/dn(x, y) = 1  for x = 0
//     du/dn(x, y) = 0  otherwise

#include <dolfin.h>
#include "Poisson.h"
  
using namespace dolfin;

int main()
{
  // Right-hand side
  class Source : public Function
  {
    real eval(const Point& p, unsigned int i)
    {
//      return p.x*sin(p.y);
      return 0.0;  
    }
  };

  // Right-hand side
  class BoundaryFlux : public Function
  {
    real eval(const Point& p, unsigned int i)
    {
      if ( std::abs(p.x) < DOLFIN_EPS)
        return 1.0;
      else
        return 0.0;  
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
  
  // Set up problem
  UnitSquare mesh(16, 16);
  Source f;
  BoundaryFlux h;
  MyBC bc;
  Poisson::BilinearForm a;
  Poisson::LinearForm L(f, h);
  PDE pde(a, L, mesh, bc);

  // Compute solution
  Function U = pde.solve();

  // Save solution to file
  File file("poisson.pvd");
  file << U;
  
  return 0;
}
