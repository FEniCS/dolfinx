// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Hoffman, 2005.
// Modified by Anders Logg, 2005.

#include <dolfin/Poisson.h>
#include <dolfin/PoissonSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
PoissonSolver::PoissonSolver(Mesh& mesh, NewFunction& f, NewBoundaryCondition& bc)
  : mesh(mesh), f(f), bc(bc)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void PoissonSolver::solve()
{
  // Define the bilinear and linear forms
  Poisson::BilinearForm a;
  Poisson::LinearForm L(f);

  // Discretize
  NewMatrix A;
  NewVector x, b;
  NewFEM::assemble(a, L, A, b, mesh, bc);

  // Solve the linear system
  NewGMRES solver;
  solver.solve(A, x, b);

  // Save function to file
  NewFunction u(mesh, x, a.trial());
  File file("poisson.m");
  file << u;
}
//-----------------------------------------------------------------------------
void PoissonSolver::solve(Mesh& mesh, NewFunction& f, NewBoundaryCondition& bc)
{
  PoissonSolver solver(mesh, f, bc);
  solver.solve();
}
//-----------------------------------------------------------------------------
