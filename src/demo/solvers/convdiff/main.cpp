// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// A simple test program for convection-diffusion, solving
//
//     du/dt + b.grad u - div a grad u = f
//
// around a hot dolphin in 2D, with diffusivity given by
//
//     a(x,y,t) = 0.1
//
// and convection given by
//
//     b(x,y,t) = (-5,0).
//
// This program illustrates the need for stabilisation, for
// instance streamline-diffusion, for large values of b. For
// |b| > 10 oscillations start to appear. Try b = (-100,0)
// to see som quite large oscillations.

#include <dolfin.h>

using namespace dolfin;

// Source term
real f(real x, real y, real z, real t)
{
  return 0.0;
}

// Diffusivity
real a(real x, real y, real z, real t)
{
  return 0.1;
}

// Convection
real b(real x, real y, real z, real t, int i)
{
  if ( i == 0 )
    return -5.0;

  return 0.0;
}

// Boundary conditions
void mybc(BoundaryCondition& bc)
{
  // u = 0 on the inflow boundary
  if ( bc.coord().x == 0.0 )
    bc.set(BoundaryCondition::NEUMANN, 0.0);
  else if ( bc.coord().x == 1.0 )
    bc.set(BoundaryCondition::DIRICHLET, 0.0);
  else if ( bc.coord().y == 0.0)
    bc.set(BoundaryCondition::NEUMANN, 0.0);
  else if ( bc.coord().y == 1.0 )
    bc.set(BoundaryCondition::NEUMANN, 0.0);
  else
    bc.set(BoundaryCondition::DIRICHLET, 1.0);
}

int main(int argc, char **argv)
{
  Mesh mesh("dolfin.xml.gz");

  /*
  Problem convdiff("convection-diffusion", mesh);

  convdiff.set("source", f);
  convdiff.set("diffusivity", a);
  convdiff.set("convection", b);
  convdiff.set("boundary condition", mybc);
  convdiff.set("final time", 0.5);
  convdiff.set("time step", 0.1);

  convdiff.solve();
  */

  return 0;
}
