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
  Function u;      // velocity;
  Function p;      // pressure; 
  Function up;     // velocity from previous time step
  Function uc;     // linearized convection velocity 
  Function f;      // force term

  real t  = 0.0;   // current time
  real T  = 1.0;   // final time
  real k  = 0.1;   // time step
  real h;          // local mesh size
  real nu = 0.01;  // viscosity 

  // stabilization parameters
  real d1,d2;      
  
  // Create variational forms
  NSEMomentum::BilinearForm a_mom(uc,k,nu,d1,d2);
  NSEMomentum::LinearForm L_mom(uc,up,f,p,k,nu,d1,d2);

  NSEContinuity::BilinearForm a_con();
  NSEContinuity::LinearForm L_con(uc,f,d1);

  Matrix Am, Ac;
  Vector xu, xuc, xup, xp, bm, bc;

  t=h+T;

  /*

  // Create variational forms
  ConvectionDiffusion::BilinearForm a(w, k, c);
  ConvectionDiffusion::LinearForm L(u0, w, f, k, c);


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







  Matrix A_mom,A_con;
  Vector x_mom, x_con, b_mom, b_con;

  // Discretize Continuity equation 
  NewFEM::assemble(a_con, L_con, A_con, b_con, mesh, element_con);

  // Set boundary conditions
  NewFEM::setBC(A_con, b_con, mesh, bc);

  while (t<T) 
  {

  // Discretize Continuity equation 
  NewFEM::assemble(L_con, b_con, mesh, element_con);

  // Discretize
  NewFEM::assemble(a, L, A, b, mesh, element);

  // Set boundary conditions
  NewFEM::setBC(A, b, mesh, bc);
  
  // Solve the linear system
  // FIXME: Make NewGMRES::solve() static
  NewGMRES solver;
  solver.solve(A, x, b);

  }


  // FIXME: Remove this and implement output for NewFunction
  Vector xold(b.size());
  for(uint i = 0; i < x.size(); i++)
    xold(i) = x(i);
  Function uold(mesh, xold);
  uold.rename("u", "temperature");
  File file("poisson.m");
  file << uold;









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
