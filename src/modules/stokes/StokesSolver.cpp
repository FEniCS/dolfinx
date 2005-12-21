// Copyright (C) 2005 Anders Logg and Andy R. Terrel.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-09-20
// Last changed: 2005-12-20

#include <dolfin/Stokes.h>
#include <dolfin/StokesSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
StokesSolver::StokesSolver(Mesh& mesh, Function& f, BoundaryCondition& bc)
  : Solver(), mesh(mesh), f(f), bc(bc)
{
  // Declare parameters
  add("velocity file name", "velocity.pvd");
  add("pressure file name", "pressure.pvd");
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
  Function w(x, mesh, a.trial());

  // Pick the two sub functions of the solution
  Function u = w[0];
  Function p = w[1];

  // Save the solutions to file
  u.rename("u", "velocity");
  p.rename("p", "pressure");
  File ufile(get("velocity file name"));
  File pfile(get("pressure file name"));
  ufile << u;
  pfile << p;
}
//-----------------------------------------------------------------------------
void StokesSolver::solve(Mesh& mesh, Function& f, BoundaryCondition& bc)
{
  StokesSolver solver(mesh, f, bc);
  solver.solve();
}
//-----------------------------------------------------------------------------
