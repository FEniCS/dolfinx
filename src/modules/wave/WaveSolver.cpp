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
  FEM::assemble(wave, mesh, A);

  StiffnessMatrix Astiff(mesh);
  MassMatrix Mmass(mesh);

  Matrix A2;
  Vector b2;


  x21(0) = -1.0;
  x21(1) = 1.0;
  x21(2) = 1.0;
  x21(3) = -1.0;                                                              

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
    FEM::assemble(wave, mesh, b);

    /*

    // dG(0)

    A2 = Astiff;
    A2 *= k * k;
    A2 += Mmass;

    Vector xtmp(x10.size());

    Mmass.mult(x10, xtmp);

    b2 = xtmp;

    Mmass.mult(x20, xtmp);
    xtmp *= k;

    b2.add(1, xtmp);

    */

    ///*

    // cG(1)

    Matrix Atmp;

    A2 = Astiff;
    A2 *= k / 2;

    Atmp = Mmass;
    Atmp *= 2 / k;

    A2 += Atmp;

    Vector xtmp(x10.size());

    Mmass.mult(x20, xtmp);
    xtmp *= 2;

    b2 = xtmp;

    Mmass.mult(x10, xtmp);
    xtmp *= 2 / k;

    b2.add(1, xtmp);

    Astiff.mult(x10, xtmp);
    xtmp *= -k / 2;

    //xtmp.show();

    b2.add(1, xtmp);

    //*/

    ///*
    // Compare assembled matrices and vectors

    cout << "A: " << endl;
    A.show();
    cout << "A2: " << endl;
    A2.show();
    cout << "b: " << endl;
    b.show();
    cout << "b2: " << endl;
    b2.show();
    //*/

    x11 = 0;
    //solver.solve(A2, x11, b2);
    solver.solve(A, x11, b);
    
    /*
    // dG(0)

    x21 = x11;
    x21.add(-1, x10);
    x21 *= 1 / k;
    */

    ///*
    // cG(1)

    x21 = 0;
    x21.add(2 / k, x11);
    x21.add(-2 / k, x10);
    x21.add(-1, x20);
    //*/


    // Save the solution
    u1.update(t);
    file << u1;

    // Update progress
    p = t / T;

  }

}
//-----------------------------------------------------------------------------
