// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// A simple test program for the Poisson solver, solving
//
//     - div grad u(x,y,z) = f(x,y,z)
//
// on the unit cube with the source f given by
//
//     f(x,y,z) = 14 pi^2 sin(pi x) sin(2pi y) sin(3pi z).
//
// and homogeneous Dirichlet boundary conditions.
// This problem has the exact solution
//
//     u(x,y,z) = sin(pi x) sin(2pi y) sin(3pi z).

#include <dolfin.h>

using namespace std;
using namespace dolfin;

// Source term
real f(real x, real y, real z, real t)
{
  real pi = DOLFIN_PI;
  return 14.0 * pi*pi * sin(pi*x) * sin(2.0*pi*y) * sin(3.0*pi*z);
}

// Boundary conditions
void mybc(BoundaryCondition& bc)
{
  bc.set(BoundaryCondition::DIRICHLET, 0.0);
}

int main()
{
  Grid grid("grid.xml.gz");
  Problem poisson("poisson", grid);

  poisson.set("source", f);
  poisson.set("boundary condition", mybc);
 
  poisson.solve();
  
  return 0;
}
