// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include "PoissonSolver.h"
#include "Poisson.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
PoissonSolver::PoissonSolver(Grid& grid) : Solver(grid)
{
  dolfin_parameter(Parameter::FUNCTION, "source", 0);
}
//-----------------------------------------------------------------------------
const char* PoissonSolver::description()
{
  return "Poisson's equation";
}
//-----------------------------------------------------------------------------
void PoissonSolver::solve()
{
  Galerkin     fem;
  Matrix       A;
  Vector       x, b;
  Function     u(grid, x);
  Function     f(grid, "source");
  Poisson      poisson(f);
  KrylovSolver solver;
  File         file("poisson.m");

  // Discretise
  fem.assemble(poisson, grid, A, b);

  
  b.show();

  // Solve the linear system
  solver.solve(A, x, b);

  x.show();

  // Save the solution
  u.rename("u", "temperature");
  file << u;
}
//-----------------------------------------------------------------------------
