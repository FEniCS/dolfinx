// Copyright (C) 2003-2006 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-01-16
// Last changed: 2006-05-07

#ifdef HAVE_PETSC_H

#include <dolfin/PoissonSolver.h>
#include <dolfin/Poisson2D.h>
#include <dolfin/Poisson3D.h>

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
  BilinearForm* a = 0;
  LinearForm* L = 0;
  if ( mesh.type() == Mesh::triangles )
  {
    dolfin_info("Solving Poisson's equation (2D).");
    a = new Poisson2D::BilinearForm();
    L = new Poisson2D::LinearForm(f);
  } 
  else if ( mesh.type() == Mesh::tetrahedra )
  {
    dolfin_info("Solving Poisson's equation (3D).");
    a = new Poisson3D::BilinearForm();
    L = new Poisson3D::LinearForm(f);
  }
  else
  {
    dolfin_error("Poisson solver only implemented for 2 and 3 space dimensions.");
  }

  // Discretize equation
  Matrix A;
  Vector x, b;
  FEM::assemble(*a, *L, A, b, mesh, bc);

  // Solve the linear system
  GMRES solver;
  solver.solve(A, x, b);

  // Save function to file
  Function u(x, mesh, a->trial());
  File file(get("solution file name"));
  file << u;

  // Delete forms
  delete a;
  delete L;
}
//-----------------------------------------------------------------------------
void PoissonSolver::solve(Mesh& mesh, Function& f, BoundaryCondition& bc)
{
  PoissonSolver solver(mesh, f, bc);
  solver.solve();
}
//-----------------------------------------------------------------------------

#endif
