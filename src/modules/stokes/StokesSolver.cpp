// Copyright (C) 2005 Anders Logg and Andy R. Terrel.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-09-20
// Last changed: 2005-09-20

#include <dolfin/Stokes.h>
#include <dolfin/StokesSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
StokesSolver::StokesSolver(Mesh& mesh, Function& f, BoundaryCondition& bc)
  : Solver(), mesh(mesh), f(f), bc(bc)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void StokesSolver::solve()
{
  // Define the bilinear and linear forms
  Stokes::BilinearForm a;
  Stokes::LinearForm L(f);

  // Discretize equation
  Matrix A;
  Vector x, b;
  FEM::assemble(a, L, A, b, mesh, bc);

  // Solve the linear system
  GMRES solver;
  solver.solve(A, x, b);

  // Save function to file
  Function u(x, mesh, a.trial());
  File file("stokes.pvd");
  file << u;
}
//-----------------------------------------------------------------------------
void StokesSolver::solve(Mesh& mesh, Function& f, BoundaryCondition& bc)
{
  StokesSolver solver(mesh, f, bc);
  solver.solve();
}
//-----------------------------------------------------------------------------
