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
		     BoundaryCondition& bc_con)
  : mesh(mesh), f(f), bc_mom(bc_mom), bc_con(bc_con) 
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NSESolver::solve()
{
  real t  = 0.0;   // current time
  real T  = 1.0;   // final time
  real k  = 0.01;   // time step
  real nu = 0.001; // viscosity 
  real C1 = 2.0;   // stabilization parameter 
  real C2 = 2.0;   // stabilization parameter 

  // Create matrices and vectors 
  Matrix Amom, Acon;
  //  Vector x0vel, xvel, xcvel, xpre, bmom, bcon;
  Vector bmom, bcon;

  int nsd = 2;

  // Initialize velocity;
  Vector x0vel(nsd*mesh.noNodes());
  Vector xcvel(nsd*mesh.noNodes());
  Vector xvel(nsd*mesh.noNodes());
  x0vel = 0.0;
  xcvel = 0.0;
  xvel = 0.0;

  Vector xpre(mesh.noNodes());
  xpre = 0.0;

  GMRES solver_con;
  GMRES solver_mom;

  cout << "Create functions" << endl;
  
  // Create functions
  Function u(xvel, mesh);   // Velocity
  Function u0(x0vel, mesh); // Velocity from previous time step 
  Function uc(xcvel, mesh); // Velocity linearized convection 
  Function p(xpre, mesh);   // Pressure

  // vectors for functions for element size and inverse of velocity norm
  Vector hvector, wnorm_vector; 
  // functions for element size and inverse of velocity norm
  Function h(hvector), wnorm(wnorm_vector);

  cout << "Create bilinear form: momentum" << endl;

  // Define the bilinear and linear forms
  NSEMomentum::BilinearForm amom(uc,wnorm,h,k,nu,C1,C2);
  NSEMomentum::LinearForm Lmom(uc,u0,f,p,wnorm,h,C1,C2,k,nu);

  cout << "Create bilinear form: continuity" << endl;

  NSEContinuity::BilinearForm acon(wnorm,h,C1);
  NSEContinuity::LinearForm Lcon(uc,f,wnorm,h,C1);
  //NSEContinuity::LinearForm Lcon(uc);

  cout << "Compute element size" << endl;

  // Compute local element size h
  ComputeElementSize(mesh, hvector);  

  cout << "Compute inverse norm" << endl;

  // Compute inverse of advective velocity norm 1/|a|
  //ConvectionNormInv(uc, wnorm, wnorm_vector);
  wnorm_vector.init(mesh.noNodes());
  wnorm_vector = 1.0;

  cout << "Create file" << endl;

  File file("pressure.m");  // file for saving pressure

  cout << "Assemble form: continuity" << endl;

  // Discretize Continuity equation 
  FEM::assemble(acon, Acon, mesh);

  // Start time-stepping
  Progress prog("Time-stepping");
  while (t<T) 
  {

    x0vel = xvel;

    for (int i=0;i<10;i++){

      xcvel = xvel;
    
      // Compute inverse of advective velocity norm 1/|a|
      // ConvectionNormInv(uc, wnorm, wnorm_vector);

      cout << "Assemble form: continuity" << endl;

      // Discretize Continuity equation 
      FEM::assemble(Lcon, bcon, mesh);

      cout << "Set boundary conditions: continuity" << endl;

      // Set boundary conditions
      FEM::applyBC(Acon, bcon, mesh, acon.trial(),bc_con);

      // Save the solution
      //p.set(t);
      //file << p;

      /*
      Acon.disp();
      bcon.disp();
      solver.setRtol(1.0e-5);
      solver.setAtol(1.0e-5);
      solver.disp();
      */
      
      //bcon.disp();
      //bcon = 0.0; 

      // Solve the linear system
      solver_con.solve(Acon, xpre, bcon);

      cout << "Assemble form: momentum" << endl;

      // Discretize Momentum equations
      FEM::assemble(amom, Lmom, Amom, bmom, mesh, bc_mom);

      cout << "Set boundary conditions: momentum" << endl;

      // Set boundary conditions
      FEM::applyBC(Amom, bmom, mesh, amom.trial(),bc_mom);

      cout << "Solve equation" << endl;

      // Solve the linear system
      solver_mom.solve(Amom, xvel, bmom);

    }


    cout << "Save solution" << endl;

    // Save the solution
    p.set(t);
    file << p;

    t = t + k;

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
		      BoundaryCondition& bc_con)
{
  NSESolver solver(mesh, f, bc_mom, bc_con);
  solver.solve();
}
//-----------------------------------------------------------------------------
void NSESolver::ComputeElementSize(Mesh& mesh, Vector& hvector)
{
  // Compute element size h
  hvector.init(mesh.noCells());	
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      hvector((*cell).id()) = (*cell).diameter();
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
  
  cout << "check" << endl;

  for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      map.update(*cell);
      wn_element.dofmap(dofs, *cell, mesh);
      wn_element.pointmap(points, components, map);
      for (uint i = 0; i < n; i++)
	{
	  convection_norm = 0.0;
	  for(uint j=0; j < m; ++j){
	    cout << "check j = " << j << endl;
	    convection_norm += pow(w(points[i], j), 2);		  
	    cout << "check" << endl;
	  }
	  wnorm_vector(dofs[i]) = 1.0 / sqrt(convection_norm);
	}
    }
  
  cout << "check" << endl;

  // Delete data
  delete [] dofs;
  delete [] components;
  delete [] points;
  
  
}
//-----------------------------------------------------------------------------
