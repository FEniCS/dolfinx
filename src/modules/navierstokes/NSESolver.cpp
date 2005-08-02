// Copyright (C) 2005 Johan Hoffman.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005
// Last changed: 2005

#include <dolfin/NSEMomentum.h>
#include <dolfin/NSEContinuity.h>
#include <dolfin/NSESolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NSESolver::NSESolver(Mesh& mesh, Function& f, BoundaryCondition& bc_mom, 
		     BoundaryCondition& bc_con, Function& u0)
  : mesh(mesh), f(f), bc_mom(bc_mom), bc_con(bc_con), u0(u0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NSESolver::solve()
{
  real t  = 0.0;   // current time
  real T  = 1.0;   // final time
  real k  = 0.1;   // time step
  real nu = 0.01;  // viscosity 
  real C1 = 1.0;   // stabilization parameter 
  real C2 = 1.0;   // stabilization parameter 

  // Create matrices and vectors 
  Matrix Amom, Acon;
  Vector xvel, x0vel, xcvel, bmom, bcon;

  GMRES solver;

  // Create functions
  Function u(xvel, mesh);   // Velocity
  Function u0(x0vel, mesh); // Velocity from previous time step 
  Function uc(xcvel, mesh); // Velocity linearized convection 
  Function p(xpre, mesh);   // Pressure

  // vectors for functions for element size and inverse of velocity norm
  Vector hvector, wnorm_vector; 
  // functions for element size and inverse of velocity norm
  Function h(hvector), wnorm(wnorm_vector);

  // Define the bilinear and linear forms
  NSEMomentum::BilinearForm amom(uc,k,nu,C1,C2);
  NSEMomentum::LinearForm Lmom(uc,up,f,p,C1,C2,k,nu);

  NSEContinuity::BilinearForm acon;
  NSEContinuity::LinearForm Lcon(uc,f,C1);

  // Compute local element size h
  ComputeElementSize(mesh, hvector);  
  // Compute inverse of advective velocity norm 1/|a|
  ConvectionNormInv(w, wnorm, wnorm_vector);


  /*
  // Set finite elements
  u.set(amom.trial());   
  up.set(amom.trial());   
  uc.set(amom.trial());   
  p.set(acon.trial());   
  */

  File file("pressure.m");  // file for saving pressure

  // Discretize Continuity equation 
  FEM::assemble(acon, Acon, mesh);

  GMRES solver;

  // Initialize velocity;
  u = u0;
  uc = u0;
  up = u0;


  // Start time-stepping
  Progress prog("Time-stepping");
  while (t<T) 
  {

    up = u;

    for (int i=0;i<1;i++){

      uc = u;

      // Discretize Continuity equation 
      FEM::assemble(Lcon, bcon, mesh);

      // Set boundary conditions
      FEM::applyBC(Acon, bcon, mesh, acon.trial(),bc_con);

      // Solve the linear system
      solver.solve(Acon, xpre, bcon);

      // Discretize Momentum equations
      FEM::assemble(amom, Lmom, Amom, bmom, mesh, bc_mom);

      // Set boundary conditions
      FEM::applyBC(Amom, bmom, mesh, amom.trial(),bc_mom);

      // Solve the linear system
      solver.solve(Amom, xvel, bmom);

    }


    // Save the solution
    p.set(t);
    file << p;

    // Update progress
    prog = t / T;
  }


  /*
  // Save function to file
  Function u(x, mesh, a.trial());
  File file("poisson.m");
  file << u;
  */


}
//-----------------------------------------------------------------------------
void NSESolver::solve(Mesh& mesh, Function& f, BoundaryCondition& bc_mom, 
		      BoundaryCondition& bc_con, Function& u0)
{
  NSESolver solver(mesh, f, bc_mom, bc_con, u0);
  solver.solve();
}
//-----------------------------------------------------------------------------
void NSESolver::ComputeElementSize(Mesh& mesh, Vector& h)
{
  // Compute element size h
  h.init(mesh.noCells());	
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      h((*cell).id()) = (*cell).diameter();
    }
}
//-----------------------------------------------------------------------------
void NSESolver::ConvectionNormInv(Function& w, Function& wnorm,
				  Vector& wnorm_vector)
{
  // Compute inverse norm of w
  const FiniteElement& wn_element = wnorm.element();
  const FiniteElement& w_element  = w.element();
  uint n = wn_element.spacedim();
  uint m = w_element.tensordim(0);
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
