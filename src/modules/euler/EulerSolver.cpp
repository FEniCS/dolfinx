// Copyright (C) 2004 Harald Svensson.
// Licensed under the GNU GPL Version 2.

#include "EulerSolver.h"
#include "Euler.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
EulerSolver::EulerSolver(Mesh& mesh) : Solver(mesh)
{
  dolfin_parameter(Parameter::REAL,      "final time",           1.0);
  dolfin_parameter(Parameter::REAL,      "time step",            0.01);
  dolfin_parameter(Parameter::VFUNCTION, "Source Momentum",     0);
  dolfin_parameter(Parameter::FUNCTION,  "Source Energy",        0);
  dolfin_parameter(Parameter::FUNCTION,  "Fluid Viscosity",      0);
  dolfin_parameter(Parameter::FUNCTION,  "Fluid Conductivity",   0);
}
//-----------------------------------------------------------------------------
const char* EulerSolver::description()
{
  return "Euler";
}
//-----------------------------------------------------------------------------
void EulerSolver::solve()
{
  Matrix Ae;
  Vector be;
  Vector ux0,ux1;
  Vector rex;

  Vector ul;

  Function::Vector u0(mesh, ux0, 5);
  Function::Vector u1(mesh, ux1, 5); 

  Function::Vector ulin(mesh, ul, 5);

  Function::Vector fm("Source Momentum",3);
  Function fe("Source Energy");
  Function am("Fluid Viscosity");
  Function ae("Fluid Conductivity");

  Function::Vector residual_energy(mesh, rex, 5);

  Euler        euler(fm,fe,am,ae,ulin,u0);

  KrylovSolver solver;

  File         file_mesh("euler.msh");
  File         file_res("euler.res");

  file_mesh << mesh;

  file_res << u0;

  real t = 0.0;
  real T = dolfin_get("final time");
  real k = dolfin_get("time step");

  // Assemble matrix
  euler.k = k;
  euler.t = t;

  cout << " Assembling A and b START   " << endl;
    
  FEM::assemble(euler, mesh, Ae, be);

  cout << " Assembling A and b END   " << endl;

  dolfin_debug("Assembled matrix:");

  // Solve the linear system
  solver.solve(Ae, ux1, be);

  // Start a progress session
  Progress p("Time-stepping");

  // Start time-stepping
  while ( t < T ) {

    // Make time step
    t += k;
    ux0 = ux1;

    // Start non linear loop
    for (int i=0; i<10; i++ ) 
    {

      // Update linearized velocity 
      ul = ux1;

      // Assemble load vector
      euler.k     = k;
      euler.t     = t;      

      FEM::assemble(euler, mesh, Ae);
      FEM::assemble(euler, mesh, be);

      // Solve the linear system
      solver.solve(Ae, ux1, be);
  
      // Check discrete residuals 
      
      Ae.mult(ux1,rex); rex -= be;

      if (( rex.norm() ) < 1.0e-2) break;
      
      cout << "l2_norm      = " << rex.norm() << endl;
    }

    // Save the solution
    for (int i=0;i<5;i++)
    {
      u1(i).update(t);
    }
    file_res << u1(0);

    // Update progress
    p = t / T;

  }
  //*/
}
//-----------------------------------------------------------------------------
