// Copyright (C) 2003 Fredrik Bengzon and Johan Jansson.
// Licensed under the GNU GPL Version 2.

#include "ElasticitySolver.h"
#include "Elasticity.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ElasticitySolver::ElasticitySolver(Mesh& mesh) : Solver(mesh)
{
  dolfin_parameter(Parameter::REAL,      "final time",  1.0);
  dolfin_parameter(Parameter::REAL,      "time step",   0.1);
  dolfin_parameter(Parameter::VFUNCTION,  "source",      0);
  dolfin_parameter(Parameter::FUNCTION,  "diffusivity", 0);
  dolfin_parameter(Parameter::VFUNCTION, "convection",  0);
}
//-----------------------------------------------------------------------------
const char* ElasticitySolver::description()
{
  return "Convection-diffusion";
}
//-----------------------------------------------------------------------------
void ElasticitySolver::solve()
{
  Matrix A;
  Vector x10, x11, x20, x21, b, xcomp;
  
  Function::Vector u0(mesh, x10, 3);
  Function::Vector u1(mesh, x11, 3);
  Function::Vector w0(mesh, x20, 3);
  Function::Vector w1(mesh, x21, 3);

  Function::Vector f("source", 3);
  Function a("diffusivity");
  Function::Vector beta("convection");
  
  Galerkin     fem;
  Elasticity   elasticity(f, u0, w0, a, beta);
  KrylovSolver solver;
  File         file("elasticity.m");
  
  real t = 0.0;
  real T = dolfin_get("final time");
  real k = dolfin_get("time step");

  // Save initial value
  //u1.rename("u", "temperature");
  //file << u1;

  // Assemble matrix
  elasticity.k = k;
  fem.assemble(elasticity, mesh, A);

  dolfin_debug("Assembled matrix:");
  //A.show();

  /*
  // Time independent

  fem.assemble(elasticity, mesh, b);
  //b.show();

  // Solve the linear system
  solver.solve(A, x11, b);
  //x1.show();
  */

  //A.show();
  //b.show();

  file << u1;


  ///*
  // Time dependent
  
  // Start a progress session
  Progress p("Time-stepping");
  
  // Start time-stepping
  while ( t < T ) {
  
    // Make time step
    t += k;
    x10 = x11;
    x20 = x21;

    //x10.show();
    
    // Assemble load vector
    elasticity.k = k;
    elasticity.t = t;
    //fem.assemble(elasticity, mesh, A);
    fem.assemble(elasticity, mesh, b);
    
    //A.show();
    //b.show();

    // Solve the linear system
    solver.solve(A, x11, b);
    //solver.solve(A, x11, b);



    x21 = x11;
    x21.add(-1, x10);
    x21 *= 1 / k;

    //x11.show();
    //x21.show();

    //x11 = x10;
    //x11.add(elasticity.k, x21);
    
    // Save the solution
    u1(0).update(t);
    //u1(0).t = t;
    file << u1;

    // Update progress
    p = t / T;

  }
  //*/
}
//-----------------------------------------------------------------------------
