// Copyright (C) 2004 Johan Hoffman. 
// Licensed under the GNU GPL Version 2.

#include "NSESolver.h"
#include "NSE_Momentum.h"
#include "NSE_Contiuity.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
NSESolver::NSESolver(Mesh& mesh) : Solver(mesh)
{
  dolfin_parameter(Parameter::REAL,      "final time",  1.0);
  dolfin_parameter(Parameter::REAL,      "time step",   0.1);
  dolfin_parameter(Parameter::FUNCTION,  "source",      0);
  dolfin_parameter(Parameter::FUNCTION,  "viscosity", 0);
}
//-----------------------------------------------------------------------------
const char* NSESolver::description()
{
  return "Navier-Stokes";
}
//-----------------------------------------------------------------------------
void NSESolver::solve()
{
  Matrix Am, Ac;
  Vector x0, x1, x2, x3, bm, bc;
  
  Function::Vector u0(mesh, x0, 3);
  Function::Vector u1(mesh, x1, 3);
  Function::Vector p1(mesh, x2, 1);
  Function::Vector ulin(mesh, x3, 3);
  Function::Vector res_mom(mesh, x4, 1);
  Function::Vector res_con(mesh, x5, 1);
  Function f("source");
  Function nu("viscosity");
  
  Galerkin      fem;
  NSE_Momentum  momentum(f, u0, a, beta);
  NSE_Contiuity continuity(f, u0, a, beta);
  KrylovSolver  solver;
  File          file("nse.m");
  
  real t = 0.0;
  real T = dolfin_get("final time");
  real k = dolfin_get("time step");

  // Save initial value
  u1.rename("u", "velocity");
  p1.rename("p", "pressure");
  file << u1;
  file << p1;

  // Assemble continuity matrix
  fem.assemble(continuity, mesh, Ac);

  // Assemble momentum matrix
  momentum.k = k;
  momentum.t = t;
  fem.assemble(momentum, mesh, Am);

  // Start a progress session
  Progress p("Time-stepping");

  // Start time-stepping
  while ( t < T ) {
    
    // Make time step
    t += k;
    x0 = x1;

    // Start non linear loop
    for (int i=0; i<10; i++ ) {

      // Update linearized velocity 
      x3 = x1;

      // Assemble continuity load vector
      fem.assemble(continuity, mesh, bc);
    
      // Solve the linear system
      solver.solve(Ac, x2, bc);
    
      // Assemble momentum load vector
      momentum.k = k;
      fem.assemble(momentum, mesh, bm);

      // Assemble momentum matrix
      fem.assemble(momentum, mesh, Am);

      // Solve the linear system
      solver.solve(Am, x1, bm);

      // Check discrete residuals 
      x4 = Am*x1-bm;
      x5 = Ac*x2-bc;
      if ((x4.norm() + x5.norm()) < 1.0e-2) break;
      
      cout << "l2_norm (mom) = " << x4.norm() << endl;
      cout << "l2_norm (con) = " << x5.norm() << endl;
    }

    // Save the solution
    u1.update(t);
    p1.update(t);
    file << u1;
    file << p1;

    // Update progress
    p = t / T;

  }

}
//-----------------------------------------------------------------------------
