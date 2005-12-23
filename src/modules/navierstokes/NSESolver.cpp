// Copyright (C) 2005 Johan Hoffman.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005.
// Modified by Anders Logg 2005.
// Modified by Johan Hoffman 2005.
//
// First added:  2005
// Last changed: 2005-12-22

#include <dolfin/timing.h>
#include <dolfin/NSEMomentum_3D.h>
#include <dolfin/NSEMomentum_2D.h>
#include <dolfin/NSEContinuity_3D.h>
#include <dolfin/NSEContinuity_2D.h>
#include <dolfin/NSESolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NSESolver::NSESolver(Mesh& mesh, Function& f, BoundaryCondition& bc_mom, 
		     BoundaryCondition& bc_con)
  : mesh(mesh), f(f), bc_mom(bc_mom), bc_con(bc_con) 
{
  // Declare parameters
  add("velocity file name", "velocity.pvd");
  add("pressure file name", "pressure.pvd");
}
//-----------------------------------------------------------------------------
void NSESolver::solve()
{
  real T0 = 0.0;        // start time 
  real t  = 0.0;        // current time
  real T  = 8.0;        // final time
  real nu = 1.0/3900.0; // viscosity 

  // Set time step (proportional to the minimum cell diameter) 
  real hmin;
  GetMinimumCellSize(mesh, hmin);  
  real k = 0.25*hmin; 

  // Create matrices and vectors 
  Matrix Amom, Acon;
  Vector bmom, bcon;

  // Get the number of space dimensions of the problem 
  int nsd = mesh.noSpaceDim();

  dolfin_info("Number of space dimensions: %d",nsd);

  // Initialize vectors for velocity and pressure 
  // x0vel: velocity from previous time step 
  // xcvel: linearized velocity 
  // xvel:  current velocity 
  // pvel:  current pressure 
  Vector x0vel(nsd*mesh.noVertices());
  Vector xcvel(nsd*mesh.noVertices());
  Vector xvel(nsd*mesh.noVertices());
  Vector xpre(mesh.noVertices());
  x0vel = 0.0;
  xcvel = 0.0;
  xvel = 0.0;
  xpre = 0.0;

  // Initialize vectors for the residuals of 
  // the momentum and continuity equations  
  Vector residual_mom(nsd*mesh.noVertices());
  Vector residual_con(mesh.noVertices());
  residual_mom = 1.0e3;
  residual_con = 1.0e3;

  // Initialize algebraic solvers 
  GMRES solver_con(Preconditioner::hypre_amg);
  GMRES solver_mom;
 
  // Create functions for the velocity and pressure 
  // (needed for the initialization of the forms)
  Function u0(x0vel, mesh); // velocity from previous time step 
  Function uc(xcvel, mesh); // velocity linearized convection 
  Function p(xpre, mesh);   // current pressure

  // Initialize stabilization parameters 
  Vector d1vector, d2vector;
  Function delta1(d1vector), delta2(d2vector);

  // Initialize the bilinear and linear forms
  BilinearForm* amom;;
  BilinearForm* acon;;
  LinearForm* Lmom;
  LinearForm* Lcon;

  if ( nsd == 3 ){
    amom = new NSEMomentum_3D::BilinearForm(uc,delta1,delta2,k,nu);
    Lmom = new NSEMomentum_3D::LinearForm(uc,u0,f,p,delta1,delta2,k,nu);
    acon = new NSEContinuity_3D::BilinearForm(delta1);
    Lcon = new NSEContinuity_3D::LinearForm(uc,f,delta1);
  } else if ( nsd == 2 ){
    amom = new NSEMomentum_2D::BilinearForm(uc,delta1,delta2,k,nu);
    Lmom = new NSEMomentum_2D::LinearForm(uc,u0,f,p,delta1,delta2,k,nu);
    acon = new NSEContinuity_2D::BilinearForm(delta1);
    Lcon = new NSEContinuity_2D::LinearForm(uc,f,delta1);
  } else{
    dolfin_error("Navier-Stokes solver only implemented for 2 and 3 space dimensions.");
  }

  // Create function for velocity 
  // (must be done after initialization of forms)
  Function u(xvel, mesh, uc.element());   // current velocity

  // Synchronise functions and boundary conditions with time
  u.sync(t);
  p.sync(t);
  bc_con.sync(t);
  bc_mom.sync(t);
    
  // Initialize output files 
  File file_u(get("velocity file name"));  // file for saving velocity 
  File file_p(get("pressure file name"));  // file for saving pressure

  // Compute stabilization parameters
  ComputeStabilization(mesh,u0,nu,k,d1vector,d2vector);

  dolfin_info("Assembling matrix: continuity");

  // Assembling matrices 
  FEM::assemble(*acon, Acon, mesh);
  FEM::assemble(*amom, Amom, mesh);

  // Initialize time-stepping parameters
  int time_step = 0;
  int sample = 0;
  int no_samples = 10;

  // Start time-stepping
  Progress prog("Time-stepping");
  while (t<T) 
  {

    time_step++;
    dolfin_info("Time step %d",time_step);

    // Set current velocity to velocity at previous time step 
    x0vel = xvel;

    // Compute stabilization parameters
    ComputeStabilization(mesh,u0,nu,k,d1vector,d2vector);

    // Initialize residuals 
    residual_mom = 1.0e3;
    residual_con = 1.0e3;

    // Fix-point iteration for non-linear problem 
    while (sqrt(sqr(residual_mom.norm()) + sqr(residual_con.norm())) > 1.0e-2){
      
      dolfin_info("Assemble vector: continuity");

      // Assemble continuity vector 
      FEM::assemble(*Lcon, bcon, mesh);

      // Set boundary conditions for continuity equation 
      FEM::applyBC(Acon, bcon, mesh, acon->trial(), bc_con);

      dolfin_info("Solve linear system: continuity");

      // Solve the linear system for the continuity equation 
      tic();
      solver_con.solve(Acon, xpre, bcon);
      dolfin_info("Linear solve took %g seconds",toc());

      dolfin_info("Assemble vector: momentum");

      // Assemble momentum vector 
      tic();
      FEM::assemble(*Lmom, bmom, mesh);
      dolfin_info("Assemble took %g seconds",toc());

      // Set boundary conditions for the momentum equation 
      FEM::applyBC(Amom, bmom, mesh, amom->trial(),bc_mom);

      dolfin_info("Solve linear system: momentum");

      // Solve the linear system for the momentum equation 
      tic();
      solver_mom.solve(Amom, xvel, bmom);
      dolfin_info("Linear solve took %g seconds",toc());
      
      // Set linearized velocity to current velocity 
      xcvel = xvel;
    
      dolfin_info("Assemble matrix and vector: momentum");
      FEM::assemble(*amom, *Lmom, Amom, bmom, mesh, bc_mom);
      FEM::applyBC(Amom, bmom, mesh, amom->trial(),bc_mom);

      // Compute residual for momentum equation 
      Amom.mult(xvel,residual_mom);
      residual_mom -= bmom;
      
      dolfin_info("Assemble vector: continuity");
      FEM::assemble(*Lcon, bcon, mesh);
      FEM::applyBC(Acon, bcon, mesh, acon->trial(),bc_con);

      // Compute residual for continuity equation 
      Acon.mult(xpre,residual_con);
      residual_con -= bcon;
      
      dolfin_info("Momentum residual  : l2 norm = %f",residual_mom.norm());
      dolfin_info("continuity residual: l2 norm = %f",residual_con.norm());
      dolfin_info("Total NSE residual : l2 norm = %f",sqrt(sqr(residual_mom.norm()) + sqr(residual_con.norm())));

    dolfin_info("save solution to file");
    file_p << p;
    file_u << u;
    }

    dolfin_info("save solution to file");
    if ( (time_step == 1) || (t > (T-T0)*(real(sample)/real(no_samples))) ){
      file_p << p;
      file_u << u;
      sample++;
    }

    // Increase time with timestep
    t = t + k;

    // Update progress
    prog = t / T;
  }

    dolfin_info("save solution to file");
    file_p << p;
    file_u << u;


  delete [] amom;;
  delete [] Lmom;;
  delete [] acon;;
  delete [] Lcon;;
}
//-----------------------------------------------------------------------------
void NSESolver::solve(Mesh& mesh, Function& f, BoundaryCondition& bc_mom, 
		      BoundaryCondition& bc_con)
{
  NSESolver solver(mesh, f, bc_mom, bc_con);
  solver.solve();
}
//-----------------------------------------------------------------------------
void NSESolver::ComputeCellSize(Mesh& mesh, Vector& hvector)
{
  // Compute cell size h
  hvector.init(mesh.noCells());	
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      hvector((*cell).id()) = (*cell).diameter();
    }
}
//-----------------------------------------------------------------------------
void NSESolver::GetMinimumCellSize(Mesh& mesh, real& hmin)
{
  // Get minimum cell diameter
  hmin = 1.0e6;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      if ((*cell).diameter() < hmin) hmin = (*cell).diameter();
    }
}
//-----------------------------------------------------------------------------
void NSESolver::ComputeStabilization(Mesh& mesh, Function& w, real nu, real k, 
				     Vector& d1vector, Vector& d2vector)
{
  // Compute least-squares stabilizing terms: 
  //
  // if  h/nu > 1 or ny < 10^-10
  //   d1 = C1 * ( 0.5 / sqrt( 1/k^2 + |U|^2/h^2 ) )   
  //   d2 = C2 * h 
  // else 
  //   d1 = C1 * h^2  
  //   d2 = C2 * h^2  

  real C1 = 4.0;   
  real C2 = 2.0;   

  d1vector.init(mesh.noCells());	
  d2vector.init(mesh.noCells());	

  real normw; 

  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    normw = 0.0;
    for (VertexIterator n(cell); !n.end(); ++n)
      normw += sqrt( sqr((w.vector())((*n).id()*2)) + sqr((w.vector())((*n).id()*2+1)) );
    normw /= (*cell).noVertices();
    if ( (((*cell).diameter()/nu) > 1.0) || (nu < 1.0e-10) ){
      d1vector((*cell).id()) = C1 * (0.5 / sqrt( 1.0/sqr(k) + sqr(normw/(*cell).diameter()) ) );
      d2vector((*cell).id()) = C2 * (*cell).diameter();
    } else{
      d1vector((*cell).id()) = C1 * sqr((*cell).diameter());
      d2vector((*cell).id()) = C2 * sqr((*cell).diameter());
    }	
  }
}
//-----------------------------------------------------------------------------

