// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include "PoissonSolverMultigrid.h"
#include "Poisson.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
PoissonSolverMultigrid::PoissonSolverMultigrid(Mesh& mesh) : Solver(mesh)
{
  dolfin_parameter(Parameter::FUNCTION, "source", 0);
}
//-----------------------------------------------------------------------------
const char* PoissonSolverMultigrid::description()
{
  return "Poisson's equation with multigrid";
}
//-----------------------------------------------------------------------------
void PoissonSolverMultigrid::solve()
{
  Vector   x;
  Function u(mesh, x);
  Function f("source");
  Poisson  poisson(f);
  File     file("poisson.m");

  cout << "Using Poisson solver with multigrid" << endl;

  // Solve the equation using multigrid
  MultigridSolver::solve(poisson, x, mesh, 2);
  
  // Save the solution
  u.rename("u", "temperature");
  file << u;
}
//-----------------------------------------------------------------------------
