// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include "ConvDiffSolver.h"
#include "ConvDiff.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ConvDiffSolver::ConvDiffSolver(Grid& grid) : Solver(grid)
{
  dolfin_parameter(Parameter::REAL,      "final time",  1.0);
  dolfin_parameter(Parameter::REAL,      "time step",   0.1);
  dolfin_parameter(Parameter::FUNCTION,  "source",      0);
  dolfin_parameter(Parameter::FUNCTION,  "diffusivity", 0);
  dolfin_parameter(Parameter::VFUNCTION, "convection",  0);
}
//-----------------------------------------------------------------------------
const char* ConvDiffSolver::description()
{
  return "Convection-diffusion";
}
//-----------------------------------------------------------------------------
void ConvDiffSolver::solve()
{
  Matrix A;
  Vector x0, x1, b;
  
  Function u0(grid, x0);
  Function u1(grid, x1);
  Function f(grid, "source");
  Function a(grid, "diffusivity");
  Function::Vector beta(grid, "convection");
  
  Galerkin     fem;
  ConvDiff     convdiff(f, u0, a, beta);
  KrylovSolver solver;
  File         file("convdiff.m");
  
  real t = 0.0;
  real T = dolfin_get("final time");
  real k = dolfin_get("time step");

  // Save initial value
  u1.rename("u", "temperature");
  file << u1;
  
  // Start a progress session
  Progress p("Time-stepping");

  // Start time-stepping
  while ( t < T ) {
    
    // Make time step
    t += k;
    
    // Set solution to previous solution
    x0 = x1;
    
    // Set time step and time for equation
    convdiff.k = k;
    convdiff.t = t;
    
    // Discretise equation
    fem.assemble(convdiff, grid, A, b);
    
    // Solve the linear system
    solver.solve(A, x1, b);
    
    // Save the solution
    u1.t = t;
    file << u1;

    // Update progress
    p = t / T;
    
  }

}
//-----------------------------------------------------------------------------
