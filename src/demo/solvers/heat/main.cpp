// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// A simple test program for the heat equation, solving
//
//     du/dt - div a grad u = f
//

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
  return 1.0;
}

// Boundary conditions
void mybc(BoundaryCondition& bc)
{
  // u = 0 on the inflow boundary
  if ( bc.coord().x == -1.5 )
    bc.set(BoundaryCondition::DIRICHLET, 0.0);
  else if ( bc.coord().y == 1.5 )
    bc.set(BoundaryCondition::DIRICHLET, 1.0);
  else
    bc.set(BoundaryCondition::NEUMANN, 0.0);
}

int main(int argc, char **argv)
{
  Mesh mesh("2particles.xml");
  Problem convdiff("convection-diffusion", mesh);

  convdiff.set("source", f);
  convdiff.set("diffusivity", a);
  convdiff.set("boundary condition", mybc);
  convdiff.set("final time", 0.5);
  convdiff.set("time step", 0.1);

  convdiff.solve();
  
  return 0;
}
