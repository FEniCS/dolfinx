// Copyright (C) 2003-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2005
//
// First added:  2003
// Last changed: 2005-12-31

#include "dolfin/ConvectionDiffusionSolver.h"
#include "dolfin/ConvectionDiffusion2D.h"
//#include "dolfin/ConvectionDiffusion3D.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ConvectionDiffusionSolver::ConvectionDiffusionSolver(Mesh& mesh, 
						     Function& w, Function& f, BoundaryCondition& bc,
                  real c, real k, real T)
  : mesh(mesh), w(w), f(f), bc(bc), c(c), k(k), T(T)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void ConvectionDiffusionSolver::solve()
{
  real t   = 0.0;  // current time

  Matrix A;              // matrix defining linear system
  Vector x0, x1, b;      // vectors 
  KrylovSolver solver(KrylovSolver::bicgstab); // linear solver
  Function u0(x0, mesh); // function at previous time step

  // File for saving solution
  File file(get("solution file name"));
  
  // Get the number of space dimensions of the problem 
  uint nsd = mesh.noSpaceDim();
  dolfin_info("Number of space dimensions: %d",nsd);

  // Vectors for functions for element size and inverse of velocity norm
  Vector hvector, wnorm_vector; 

  // Functions for element size and inverse of velocity norm
  Function h(hvector), wnorm(wnorm_vector);

  // Create variational forms
  BilinearForm* a =0;
  LinearForm* L =0;
  if( nsd == 2 )
  {  
    a = new ConvectionDiffusion2D::BilinearForm(w, wnorm, h, k, c);
    L = new ConvectionDiffusion2D::LinearForm(u0, w, wnorm, f, h, k, c);
  } 
  else if ( nsd == 3 )
  {
    dolfin_error("3D convection-diffusion is currently disabled to limit compile time.");
//    a = new ConvectionDiffusion3D::BilinearForm(w, wnorm, h, k, c);
//    L = new ConvectionDiffusion3D::LinearForm(u0, w, wnorm, f, h, k, c);
  }
  else
  {
    dolfin_error("Convection-diffusion solver only implemented for 2 and 3 space dimensions.");
  }

  // Compute local element size h
  ComputeElementSize(mesh, hvector);  

  // Compute stabiliation term  tau/2|a|
  // It is assumed that the advective velocity can be prepresnted using a linear basis
  ConvectionNormInv(w, wnorm_vector, nsd);

  // Assemble stiffness matrix
  FEM::assemble(*a, A, mesh);

  uint N = FEM::size(mesh, a->trial());
  x1.init(N);
  Function u1(x1, mesh, a->trial());
  
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
    FEM::assemble(*L, b, mesh);
    FEM::applyBC(A, b, mesh, a->trial(), bc);
    
    // Solve the linear system
    solver.solve(A, x1, b);
    
    // Save the solution to file
    file << u1;

    // Update progress
    p = t / T;
  }
  
  delete a;
  delete L;
}
//-----------------------------------------------------------------------------
void ConvectionDiffusionSolver::solve(Mesh& mesh, Function& w, Function& f, 
                            BoundaryCondition& bc, real c, real k, real T)
{
  ConvectionDiffusionSolver solver(mesh, w, f, bc, c, k, T);
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
void ConvectionDiffusionSolver::ConvectionNormInv(Function& w, Vector& wnorm_vector, uint nsd)
{
  real tau = 1.0;  // stabilisation parameter

  real norm;
  wnorm_vector.init(mesh.noVertices());
  for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
  {
    norm = 0.0;
    for (uint i =0; i < nsd; ++i)
      norm += w(*vertex, i)*w(*vertex, i);

    norm = 0.5*tau/sqrt(norm);
    wnorm_vector((*vertex).id()) = norm;  
  }

}
//-----------------------------------------------------------------------------
