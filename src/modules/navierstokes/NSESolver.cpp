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
  real h;          // local mesh size
  real nu = 0.01;  // viscosity 

  // stabilization parameters
  real d1,d2;      

  // FIXME: remove this!
  d1 = d2 = h = 1.0;
  
  // Create matrices and vectors 
  Matrix Amom, Acon;
  Vector xvel, xpvel, xcvel, xpre, bmom, bcon;

  // Create functions
  Function u(xvel, mesh);   // Velocity
  Function up(xpvel, mesh); // Velocity from previous time step 
  Function uc(xcvel, mesh); // Velocity linearized convection 
  Function p(xpre, mesh);   // Pressure

  // Define the bilinear and linear forms
  NSEMomentum::BilinearForm amom(uc,k,nu,d1,d2);
  NSEMomentum::LinearForm Lmom(uc,up,f,p,k,nu,d1,d2);

  NSEContinuity::BilinearForm acon;
  NSEContinuity::LinearForm Lcon(uc,f,d1);

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
