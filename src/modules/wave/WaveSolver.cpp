// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include "WaveSolver.h"
#include "Wave.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
WaveSolver::WaveSolver(Mesh& mesh) : Solver(mesh)
{
  dolfin_parameter(Parameter::REAL,      "final time",  1.0);
  dolfin_parameter(Parameter::REAL,      "time step",   0.1);
  dolfin_parameter(Parameter::FUNCTION,  "source",      0);
}
//-----------------------------------------------------------------------------
const char* WaveSolver::description()
{
  return "Wave";
}
//-----------------------------------------------------------------------------
void WaveSolver::solve()
{
  Matrix A;
  Vector x10, x11, x20, x21, b;
  
  Function u0(mesh, x10);
  Function u1(mesh, x11);
  Function w0(mesh, x20);
  Function w1(mesh, x21);
  Function f("source");
  
  Galerkin     fem;
  Wave         wave(f, u0, w0);
  KrylovSolver solver;
  File         file("wave.m");
  
  real t = 0.0;
  real T = dolfin_get("final time");
  real k = dolfin_get("time step");

  //x11(1) = 1.0;
  //x11(2) = 1.0;

  // Save initial value
  u1.rename("u", "temperature");
  file << u1;

  // Assemble matrix
  wave.k = k;
  fem.assemble(wave, mesh, A);

  //A.show();

  // Start a progress session
  Progress p("Time-stepping");

  // Start time-stepping
  while ( t < T ) {
    
    // Make time step
    t += k;
    x10 = x11;
    x20 = x21;
    
    //dolfin_debug("x10:");
    //x10.show();
    //dolfin_debug("x20:");
    //x20.show();

    // Assemble load vector
    wave.k = k;
    wave.t = t;
    fem.assemble(wave, mesh, b);

    //b.show();

    // Solve the linear system
    //solver.solve(A, x21, b);
    solver.solve(A, x11, b);
    
    //x11.show();

    //dolfin_debug("A:");

    //A.show();

    //dolfin_debug("b:");

    //b.show();

    //x11.show();

    x21 = x11;
    x21.add(-1, x10);
    x21 *= 1 / k;

    //x21 = x11;
    //x21.add(-1, x10);
    //x21 *= 2 / k;
    //x21.add(-1, x20);

    //x21.show();

    // Save the solution
    //u1.t = t;
    u1.update(t);
    file << u1;

    // Update progress
    p = t / T;

  }

}
//-----------------------------------------------------------------------------
