// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>
#include "Poisson.h"
#include "NewPoisson.h"
#include "PoissonSystem.h"

using namespace dolfin;

// Source term
real source(real x, real y, real z, real t)
{
  real pi = DOLFIN_PI;
  return 14.0 * pi*pi * sin(pi*x) * sin(2.0*pi*y) * sin(3.0*pi*z);
}

// Boundary conditions
void mybc(BoundaryCondition& bc)
{
  bc.set(BoundaryCondition::DIRICHLET, 0.0);
}

// Test old assembly
real testOld(Mesh& mesh, File& file)
{
  // Create variational formulation
  Function f(source);
  Poisson poisson(f);
  
  // Assemble system
  Matrix A;
  Vector b;
  tic();
  FEM::assemble(poisson, mesh, A, b);
  real t = toc();

  // Solve system
  Vector x;
  KrylovSolver solver;
  solver.solve(A, x, b);

  // Save solution
  Function u(mesh, x);
  u.rename("u1", "temperature");
  file << u;

  // Save system
  A.rename("A1", "matrix");
  b.rename("b1", "vector");
  file << A;
  file << b;

  return t;
}

// Test new assembly
real testNew(Mesh& mesh, File& file)
{
  // Create variational formulation
  Function f(source);
  NewPoisson poisson(f);
  
  // Assemble system
  Matrix A;
  Vector b;
  tic();
  NewFEM::assemble(poisson, mesh, A, b);
  real t = toc();

  // Solve system
  Vector x;
  KrylovSolver solver;
  solver.solve(A, x, b);

  // Save solution
  Function u(mesh, x);
  u.rename("u2", "temperature");
  file << u;

  // Save system
  A.rename("A2", "matrix");
  b.rename("b2", "vector");
  file << A;
  file << b;

  return t;
}

int main()
{
  dolfin_set("output", "plain text");
  dolfin_set("boundary condition", mybc);

  Mesh mesh("mesh.xml.gz");
  File file("poisson.m");
  
  dolfin_log(false);
  
  real t1 = testOld(mesh, file);
  real t2 = testNew(mesh, file);

  dolfin_log(true);

  cout << "Old assembly: " << t1 << endl;
  cout << "New assembly: " << t2 << endl;

  return 0;
}
