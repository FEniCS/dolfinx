// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Hoffman, 2005.
// Modified by Anders Logg, 2005.

#include "dolfin/ConvectionDiffusionSolver.h"
#include "dolfin/ConvectionDiffusion.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ConvectionDiffusionSolver::ConvectionDiffusionSolver(Mesh& mesh, 
						     Function& w,
						     Function& f,
						     BoundaryCondition& bc)
  : mesh(mesh), w(w), f(f), bc(bc)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void ConvectionDiffusionSolver::solve()
{
  real t = 0.0;  // current time
  real T = 1.0;  // final time
  real k = 0.1;  // time step
  real c = 0.1;  // diffusion

  Matrix A;                 // matrix defining linear system
  Vector x0, x1, b;         // vectors 
  GMRES solver;             // linear system solver
  Function u0(x0, mesh);    // function at left end-point
  File file("convdiff.m");  // file for saving solution

  // Create variational forms
  ConvectionDiffusion::BilinearForm a(w, k, c);
  ConvectionDiffusion::LinearForm L(u0, w, f, k, c);

  // Assemble stiffness matrix
  FEM::assemble(a, A, mesh);

  // FIXME: Temporary fix
  x1.init(mesh.noNodes());
  Function u1(x1, mesh, a.trial());

  // Start time-stepping
  Progress p("Time-stepping");
  while ( t < T )
  {
    // Make time step
    t += k;
    x0 = x1;
    
    // Assemble load vector and set boundary conditions
    FEM::assemble(L, b, mesh);
    FEM::setBC(A, b, mesh, a.trial(), bc);
    
    // Solve the linear system
    solver.solve(A, x1, b);
    
    // Save the solution
    u1.set(t);
    file << u1;

    // Update progress
    p = t / T;
  }
}
//-----------------------------------------------------------------------------
void ConvectionDiffusionSolver::solve(Mesh& mesh, Function& w, Function& f,
				      BoundaryCondition& bc)
{
  ConvectionDiffusionSolver solver(mesh, w, f, bc);
  solver.solve();
}
//-----------------------------------------------------------------------------
