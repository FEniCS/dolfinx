// Copyright (C) 2003 Fredrik Bengzon and Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2004.

#include "ElasticityStationarySolver.h"
#include "ElasticityStationary.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ElasticityStationarySolver::ElasticityStationarySolver(Mesh& mesh) : Solver(mesh)
{
  dolfin_parameter(Parameter::REAL,      "final time",  1.0);
  dolfin_parameter(Parameter::REAL,      "time step",   0.1);
  dolfin_parameter(Parameter::VFUNCTION,  "source",      0);
  dolfin_parameter(Parameter::FUNCTION,  "diffusivity", 0);
  dolfin_parameter(Parameter::VFUNCTION, "convection",  0);
}
//-----------------------------------------------------------------------------
const char* ElasticityStationarySolver::description()
{
  return "Stationary Elasticity";
}
//-----------------------------------------------------------------------------
void ElasticityStationarySolver::solve()
{
  Matrix A;
  Vector x10, x11, x20, x21, b, xcomp;
  
  Function::Vector u0(mesh, x10, 3);
  Function::Vector u1(mesh, x11, 3);
  Function::Vector w0(mesh, x20, 3);
  Function::Vector w1(mesh, x21, 3);

  Function::Vector f("source", 3);
  
  ElasticityStationary   elasticity(f, u0, w0);
  KrylovSolver solver;
  File         file("elasticitystationary.m");
  
  // Time independent

  // Assemble matrix and vector
  FEM::assemble(elasticity, mesh, A, b);

  // Solve the linear system
  //solver.setMethod(KrylovSolver::CG);
  //solver.setPreconditioner(KrylovSolver::ILU0);
  solver.solve(A, x11, b);
  //x1.show();

  //A.show();
  //b.show();

  //cout << "dot(b, b): " << (b * b) << endl; 
  //cout << "dot(x, x): " << (x11 * x11) << endl; 
  //printf("dot(b, b): %30.30lf\n", (b * b));
  //printf("dot(x, x): %30.30lf\n", (x11 * x11));

  file << u1;
}
//-----------------------------------------------------------------------------
