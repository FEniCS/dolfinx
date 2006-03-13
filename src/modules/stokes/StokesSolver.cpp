// Copyright (C) 2005-2006 Anders Logg and Andy R. Terrel.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-09-20
// Last changed: 2006-01-09

#include <dolfin/L2Error.h>
#include <dolfin/Stokes2D.h>
#include <dolfin/Stokes3D.h>
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
  BilinearForm* a = 0;
  LinearForm* L = 0;
  if ( mesh.type() == Mesh::triangles )
  {
    dolfin_info("Solving the Stokes equations (2D).");
    a = new Stokes2D::BilinearForm();
    L = new Stokes2D::LinearForm(f);
  } 
  else if ( mesh.type() == Mesh::tetrahedra )
  {
    dolfin_info("Solving the Stokes equations (3D).");
    a = new Stokes3D::BilinearForm();
    L = new Stokes3D::LinearForm(f);
  }
  else
  {
    dolfin_error("Stokes solver only implemented for 2 and 3 space dimensions.");
  }

  // Discretize equation
  Matrix A;
  Vector x, b;
  FEM::assemble(*a, *L, A, b, mesh, bc);

  // Solve the linear system
  GMRES solver;
  solver.solve(A, x, b);
  Function w(x, mesh, a->trial());

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
  if ( mesh.type() == Mesh::triangles )
    checkError(mesh, u);

  // Delete forms
  delete a;
  delete L;
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
