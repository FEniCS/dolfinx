// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include "PoissonSolver.h"
#include "Poisson.h"

using namespace dolfin;

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
  //Function     u(grid, x);
  Function     f(grid, "source");
  Poisson      equation(f);
  KrylovSolver solver;

  // Discretise
  tic();
  fem.assemble(equation, grid, A, b);
  toc();
  
  File file("matrix.m");
  file << A;
  
  // Solve the linear system
  solver.solve(A, x, b);

  // Save the solution
  //u.setLabel("u","temperature");
  //u.save();
}
//-----------------------------------------------------------------------------
