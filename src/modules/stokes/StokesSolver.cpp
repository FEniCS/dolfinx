// Copyright (C) 2005 Anders Logg and Andy R. Terrel.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-09-20
// Last changed: 2005-12-30

#include <dolfin/L2Error.h>
#include <dolfin/Stokes.h>
#include <dolfin/StokesSolver.h>

using namespace dolfin;

class ExactSolution : public Function
{
  real eval(const Point& p, unsigned int i)
  {
    if ( i == 0 )
      return - sin(DOLFIN_PI*p.x) * cos(DOLFIN_PI*p.y);
    else
      return cos(DOLFIN_PI*p.x) * sin(DOLFIN_PI*p.y);
  }
};

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
  solver.setRtol(1.0e-15);
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

  // Temporary for testing
  checkError(mesh, u);
}
//-----------------------------------------------------------------------------
void StokesSolver::solve(Mesh& mesh, Function& f, BoundaryCondition& bc)
{
  StokesSolver solver(mesh, f, bc);
  solver.solve();
}
//-----------------------------------------------------------------------------
void StokesSolver::checkError(Mesh& mesh, Function& U)
{
  ExactSolution u;
  L2Error::LinearForm L(U, u);
  Vector b;
  FEM::assemble(L, b, mesh);
  real norm = sqrt(b.sum());

  dolfin_info("L2 error for velocity: %.3e", norm);
}
//-----------------------------------------------------------------------------
