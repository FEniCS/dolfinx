// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include "WaveVectorSolver.h"
#include "WaveVector.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
WaveVectorSolver::WaveVectorSolver(Mesh& mesh) : Solver(mesh)
{
  dolfin_parameter(Parameter::REAL,      "final time",  1.0);
  dolfin_parameter(Parameter::REAL,      "time step",   0.1);
  dolfin_parameter(Parameter::VFUNCTION,  "source",      0);
}
//-----------------------------------------------------------------------------
const char* WaveVectorSolver::description()
{
  return "WaveVector";
}
//-----------------------------------------------------------------------------
void WaveVectorSolver::solve()
{
  Matrix A;
  Vector x10, x11, x20, x21, b;
  
  Function::Vector u0(mesh, x10, 2);
  Function::Vector u1(mesh, x11, 2);
  Function::Vector w0(mesh, x20, 2);
  Function::Vector w1(mesh, x21, 2);
  Function::Vector f("source", 2);
  
  WaveVector   wavevector(f, u0, w0);
  KrylovSolver solver;
  File         file("wavevector.m");
  
  real t = 0.0;
  real T = dolfin_get("final time");
  real k = dolfin_get("time step");

  //x11(1) = 1.0;
  //x11(2) = 1.0;

  // Save initial value
  u1(0).rename("u", "temperature");
  file << u1;

  // Assemble matrix
  wavevector.k = k;
  FEM::assemble(wavevector, mesh, A);

  // Start a progress session
  Progress p("Time-stepping");

  // Start time-stepping
  while ( t < T ) {
    
    // Make time step
    t += k;
    x10 = x11;
    x20 = x21;

    // Assemble load vector
    wavevector.k = k;
    wavevector.t = t;
    FEM::assemble(wavevector, mesh, b);

    // Solve the linear system
    //solver.solve(A, x21, b);
    solver.solve(A, x11, b);

    //dolfin_debug("A:");


    //dolfin_debug("b:");

    x21 = x11;
    x21.add(-1, x10);
    x21 *= 1 / k;

    //x21 = x11;
    //x21.add(-1, x10);
    //x21 *= 2 / k;
    //x21.add(-1, x20);

    // Save the solution
    //u1(0).t = t;
    u1(0).update(t);
    file << u1;

    // Update progress
    p = t / T;

  }

}
//-----------------------------------------------------------------------------
