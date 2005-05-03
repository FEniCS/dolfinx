// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Hoffman, 2005.
// Modified by Anders Logg, 2005.

#include <dolfin/Poisson.h>
#include <dolfin/PoissonSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
PoissonSolver::PoissonSolver(Mesh& mesh, Function& f, BoundaryCondition& bc)
  : Solver(), mesh(mesh), f(f), bc(bc)
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
  Matrix A;
  Vector x, b;
  FEM::assemble(a, L, A, b, mesh, bc);

  // Solve the linear system
  GMRES solver;
  solver.solve(A, x, b);

  // Save function to file
  Function u(x, mesh, a.trial());
  File file("poisson.m");
  file << u;
}
//-----------------------------------------------------------------------------
void PoissonSolver::solve(Mesh& mesh, Function& f, BoundaryCondition& bc)
{
  PoissonSolver solver(mesh, f, bc);
  solver.solve();
}
//-----------------------------------------------------------------------------
