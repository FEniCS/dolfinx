// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include "PoissonSolver.h"
#include "Poisson.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
PoissonSolver::PoissonSolver(Mesh& mesh) : Solver(mesh)
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
  Matrix       A;
  Vector       x, b;
  Function     u(mesh, x);
  Function     f("source");
  Poisson      poisson(f);
  KrylovSolver solver;
  File         file("poisson.m");

  // Discretise
  FEM::assemble(poisson, mesh, A, b);

  // Solve the linear system
  solver.solve(A, x, b);

  // Save the solution
  u.rename("u", "temperature");
  file << u;
}
//-----------------------------------------------------------------------------
