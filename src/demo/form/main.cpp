// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>
#include "Poisson.h" 
#include "OptimizedPoisson.h"
#include "FFCPoisson.h"

using namespace dolfin;

#define N 10 // Number of times to do the assembly

// Test old assembly
real testOld(Mesh& mesh)
{
  cout << "Testing old assembly..." << endl;

  Poisson poisson;
  Matrix A;
  tic();
  for(unsigned int i = 0; i < N; i++)
  {
    A = 0.0;
    FEM::assemble(poisson, mesh, A);
  }

  return toc();
}

// Test new assembly (hand-optimized)
real testOptimized(Mesh& mesh)
{
  cout << "Testing new assembly, hand-optimized..." << endl;

  OptimizedPoissonFiniteElement element;
  OptimizedPoissonBilinearForm a(element);
  Matrix A;
  tic();
  for(unsigned int i = 0; i < N; i++)
  {
    A = 0.0;
    NewFEM::assemble(a, mesh, A);
  }

  return toc();
}

// Test new assembly (FFC)
real testFFC(Mesh& mesh)
{
  cout << "Testing new assembly, FFC..." << endl;

  FFCPoissonFiniteElement element;
  FFCPoissonBilinearForm a(element);
  Matrix A;
  tic();
  for(unsigned int i = 0; i < N; i++)
  {
    A = 0.0;
    NewFEM::assemble(a, mesh, A);
  }

  return toc();
}

// Test new assembly (FFC) with PETSc
real testFFCPETSc(Mesh& mesh)
{
  cout << "Testing new assembly, FFC with PETSc, not doing anything..." << endl;

  tic();
  
  FFCPoissonFiniteElement element;
  FFCPoissonBilinearForm a(element);
  NewFEM::testPETSc(a, mesh);

  return toc();
}

int main()
{
  dolfin_set("output", "plain text");

  Mesh mesh("mesh.xml.gz");
  mesh.refineUniformly();
  mesh.refineUniformly();
  
  dolfin_log(false);
  
  real t1 = testOld(mesh);
  real t2 = testOptimized(mesh);
  real t3 = testFFC(mesh);
  real t4 = testFFCPETSc(mesh);

  dolfin_log(true);

  cout << "Old assembly:   " << t1 << endl;
  cout << "New, optimized: " << t2 << endl;
  cout << "New, FFC:       " << t3 << endl;
  cout << "New, FFC/PETSc: " << t4 << endl;

  return 0;
}
