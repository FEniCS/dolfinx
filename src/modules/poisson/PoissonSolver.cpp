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
  Poisson      eq;
  Galerkin     fem;
  Matrix       A;
  Vector       x, b;
  Function     u(grid, x);
  Function     f(grid, "source");
  KrylovSolver solver;

  // Discretise
  fem.assemble(equation, grid, A, b);
  
  // Solve the linear system
  solver.solve(A, x, b);
    
  // Save the solution
  u.setLabel("u","temperature");
  u.save();
}
//-----------------------------------------------------------------------------
