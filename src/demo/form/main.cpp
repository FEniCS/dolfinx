// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>
#include "Poisson.h"
#include "PoissonSystem.h"

using namespace dolfin;

// Source term
real source(real x, real y, real z, real t)
{
  real pi = DOLFIN_PI;
  return 14.0 * pi*pi * sin(pi*x) * sin(2.0*pi*y) * sin(3.0*pi*z);
}

int main()
{
  dolfin_set("output", "plain text");

  // Create variational formulation
  Function f(source);
  Poisson poisson(f);

  // Assemble system
  Mesh mesh("mesh.xml.gz");
  Matrix A;
  Vector b;
  NewFEM::assemble(poisson, mesh, A, b);
  
  // Solve system
  Vector x;
  KrylovSolver solver;
  solver.solve(A, x, b);

  // Save solution
  File file("poisson.m");
  Function u(mesh, x);
  u.rename("u", "temperature");
  file << u;
  
  return 0;
}
