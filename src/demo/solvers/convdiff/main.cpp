// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.
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

// Convection
class Convection : public NewFunction
{
  real operator() (const Point& p, unsigned int i) const
  {
    if ( i == 0 )
      return -5.0;
    else
      return 0.0;
  }
};

// Right-hand side
class Source : public NewFunction
{
  real operator() (const Point& p) const
  {
    return 0.0;
  }
};

// Boundary condition
class MyBC : public NewBoundaryCondition
{
  const BoundaryValue operator() (const Point& p)
  {
    BoundaryValue value;

    if ( p.x == 1.0 )
      value.set(0.0);
    else if ( p.x != 0.0 && p.x != 1.0 && p.y != 0.0 && p.y != 1.0 )
      value.set(1.0);
    
    return value;
  }
};

int main(int argc, char **argv)
{
  Mesh mesh("dolfin.xml.gz");
  Convection w;
  Source f;
  MyBC bc;
  
  ConvectionDiffusionSolver::solve(mesh, w, f, bc);
  
  return 0;
}
