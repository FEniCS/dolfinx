// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Hoffman, 2005.
// Modified by Anders Logg, 2005.

#include "Poisson.h"
#include "PoissonSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
PoissonSolver::PoissonSolver(Mesh& mesh) : Solver(mesh)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const char* PoissonSolver::description()
{
  return "Poisson's equation";
}
//-----------------------------------------------------------------------------
void PoissonSolver::solve()
{
  Poisson::FiniteElement element;
  Poisson::BilinearForm a(element);
  Poisson::LinearForm L(element);
  //NewMatrix A;
  //NewVector x, b;

  //b.init(A.size(0));
  //b = 1.0;

  //  NewFunction  u(mesh, x);
  //  NewFunction  f("source");
  //  NewGMRES     solver;
  //File         file("poisson.m");

  // Discretise
  //NewFEM::assemble(a, L, mesh, A, b);
  //NewFEM::assemble(a, mesh, A);

  // Solve the linear system
  //solver.solve(A, x, b);

  // Save the solution
  //u.rename("u", "temperature");
  //file << u;
}
//-----------------------------------------------------------------------------
