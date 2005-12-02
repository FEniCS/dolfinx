// Copyright (C) 2003-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2005
//
// First added:  2003
// Last changed: 2005-12-01

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
  real t   = 0.0;  // current time
  real T   = 0.3;  // final time
  real k   = 0.01;  // time step
  real c   = 0.05;  // diffusion
  real tau = 1.0;  // stabilisation parameter

  Matrix A;                   // matrix defining linear system
  Vector x0, x1, b;           // vectors 
  GMRES solver;               // linear system solver
  Function u0(x0, mesh);      // function at left end-point
  File file("convdiff.pvd");  // file for saving solution

  // vectors for functions for element size and inverse of velocity norm
  Vector hvector, wnorm_vector; 
  // functions for element size and inverse of velocity norm
  Function h(hvector), wnorm(wnorm_vector);

  // Create variational forms
  ConvectionDiffusion::BilinearForm a(w, wnorm, h, k, c, tau);
  ConvectionDiffusion::LinearForm L(u0, w, wnorm, f, h, k, c, tau);

  // Compute local element size h
  ComputeElementSize(mesh, hvector);  
  // Compute inverse of advective velocity norm 1/|a|
  ConvectionNormInv(w, wnorm, wnorm_vector);

  // Assemble stiffness matrix
  FEM::assemble(a, A, mesh);

  // FIXME: Temporary fix
  x1.init(mesh.noVertices());
  Function u1(x1, mesh, a.trial());
  
  // Synchronize function u1 with time t
  u1.sync(t);

  // Start time-stepping
  Progress p("Time-stepping");
  while ( t < T )
  {
    // Make time step
    t += k;
    x0 = x1;
    
    // Assemble load vector and set boundary conditions
    FEM::assemble(L, b, mesh);
    FEM::applyBC(A, b, mesh, a.trial(), bc);
    
    // Solve the linear system
    solver.solve(A, x1, b);
    
    // Save the solution
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
void ConvectionDiffusionSolver::ComputeElementSize(Mesh& mesh, Vector& h)
{
  // Compute element size h
  h.init(mesh.noCells());	
	for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    h((*cell).id()) = (*cell).diameter();
  }
}
//-----------------------------------------------------------------------------
void ConvectionDiffusionSolver::ConvectionNormInv(Function& w, Function& wnorm,
						  Vector& wnorm_vector)
{
  // Compute inverse norm of w
  const FiniteElement& wn_element = wnorm.element();
  uint n = wn_element.spacedim();
  uint m = w.vectordim();
  int* dofs = new int[n];
  uint* components = new uint[n];
  Point* points = new Point[n];
  AffineMap map;
  real convection_norm;
	
  wnorm_vector.init(mesh.noCells()*wn_element.spacedim());	
  
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    map.update(*cell);
    wn_element.dofmap(dofs, *cell, mesh);
    wn_element.pointmap(points, components, map);
    for (uint i = 0; i < n; i++)
    {
      convection_norm = 0.0;
      for(uint j=0; j < m; ++j) convection_norm += pow(w(points[i], j), 2);		  
      wnorm_vector(dofs[i]) = 1.0 / sqrt(convection_norm);
    }
  }
  
  // Delete data
  delete [] dofs;
  delete [] components;
  delete [] points;
}
//-----------------------------------------------------------------------------
