// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include "HeatSolver.h"
#include "Heat.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
HeatSolver::HeatSolver(Mesh& mesh) : Solver(mesh)
{
  dolfin_parameter(Parameter::REAL,      "final time",  1.0);
  dolfin_parameter(Parameter::REAL,      "time step",   0.1);
  dolfin_parameter(Parameter::FUNCTION,  "source",      0);
  dolfin_parameter(Parameter::FUNCTION,  "diffusivity", 1.0);
}
//-----------------------------------------------------------------------------
const char* HeatSolver::description()
{
  return "Heat";
}
//-----------------------------------------------------------------------------
void HeatSolver::solve()
{
  Matrix A;
  Vector x0, x1, b;
  
  Function u0(mesh, x0);
  Function u1(mesh, x1);
  Function f("source");
  Function a("diffusivity");
  
  Galerkin     fem;
  Heat         heat(f, u0, a);
  KrylovSolver solver;
  File         file("heat.dx");
  
  real t = 0.0;
  real T = dolfin_get("final time");
  real k = dolfin_get("time step");

  // Save initial value
  u1.rename("u", "temperature");
  file << mesh;
  // file << u1;

  // Assemble matrix
  heat.k = k;
  fem.assemble(heat, mesh, A);

  // Start a progress session
  Progress p("Time-stepping");

  // Start time-stepping
  while ( t < T ) {
    
    // Make time step
    t += k;
    x0 = x1;
    
    // Assemble load vector
    heat.k = k;
    heat.t = t;
    fem.assemble(heat, mesh, b);
    
    // Solve the linear system
    solver.solve(A, x1, b);
    
    // Save the solution
    u1.update(t);
    // file << u1;

    // Update progress
    p = t / T;

  }

  // file << u1;

}
//-----------------------------------------------------------------------------
