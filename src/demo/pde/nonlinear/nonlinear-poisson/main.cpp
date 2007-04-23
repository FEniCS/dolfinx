// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2005
//
// First added:  2005
// Last changed: 2006-03-01
//
// This program illustrates the use of the DOLFIN for solving a nonlinear PDE
// by solving the nonlinear variant of Poisson's equation
//
//     - div (1+u^2) grad u(x, y) = f(x, y)
//
// on the unit square with source f given by
//
//     f(x, y) = t * x * sin(y)
//
// and boundary conditions given by
//
//     u(x, y)     = t  for x = 0
//     du/dn(x, y) = 0  otherwise
//
// where t is pseudo time.
//
// This is equivalent to solving: 
// F(u) = (grad(v), (1-u^2)*grad(u)) - f(x,y) = 0
//

#include <dolfin.h>
#include "NonlinearPoisson.h"
  
using namespace dolfin;

// Source term
class MyFunction : public Function
{
  real eval(const Point& p, unsigned int i)
  {
    return time()*p.x()*sin(p.y());
  }
};

// Boundary conditions
class MyBC : public BoundaryCondition
{
  void eval(BoundaryValue& value, const Point& p, unsigned int i)
  {
    if ( std::abs(p.x() - 1.0) < DOLFIN_EPS )
      value = time();
  }
};

int main(int argc, char* argv[])
{
  dolfin_init(argc, argv);
 
  // Set up problem
  UnitSquare mesh(16, 16);
  MyFunction f;
  MyBC bc;
  Function U;

  // Create forms and nonlinear PDE
  NonlinearPoisson::BilinearForm a(U);
  NonlinearPoisson::LinearForm L(U, f);
  PDE pde(a, L, mesh, bc, PDE::nonlinear);

  // Solve nonlinear problem in a series of steps
  real dt = 1.0; real t  = 0.0; real T  = 3.0;
  f.sync(t);
  bc.sync(t);

  pde.set("Newton relative tolerance", 1e-6); 
  pde.set("Newton convergence criterion", "incremental"); 
//  pde.set("Newton convergence criterion", "residual"); 
  while( t < T)
  {
    t += dt;
    pde.solve(U);
  }

  // Save function to file
  File file("nonlinear_poisson.pvd");
  file << U;

  return 0;
}
