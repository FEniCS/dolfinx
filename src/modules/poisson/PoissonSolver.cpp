// Copyright (C) 2003-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-01-16
// Last changed: 2005-12-08

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

  // Discretize equation
  Matrix A;
  Vector x, b;
  FEM::assemble(a, L, A, b, mesh, bc);

  // Solve the linear system
  GMRES solver;
  solver.solve(A, x, b);

  // Save function to file
  Function u(x, mesh, a.trial());
  File file("poisson.pvd");
  file << u;
}
//-----------------------------------------------------------------------------
void PoissonSolver::solve(Mesh& mesh, Function& f, BoundaryCondition& bc)
{
  PoissonSolver solver(mesh, f, bc);
  solver.solve();
}
//-----------------------------------------------------------------------------
