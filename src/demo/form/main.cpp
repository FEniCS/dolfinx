// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>
#include "Poisson.h" 
#include "OptimizedPoisson.h"
#include "FFCPoisson.h"

using namespace dolfin;

#define N 2 // Number of times to do the assembly
#define M 3 // Number of times to refine the mesh 

// Test old assembly
real testOld(Mesh& mesh)
{
  cout << "--- Testing old assembly ---" << endl;

  Poisson poisson;
  Matrix A;
  tic();
  for (unsigned int i = 0; i < N; i++)
    FEM::assemble(poisson, mesh, A);

  return toc();
}

// Test new assembly (hand-optimized)
real testOptimized(Mesh& mesh)
{
  cout << "--- Testing new assembly, hand-optimized ---" << endl;

  OptimizedPoissonFiniteElement element;
  OptimizedPoissonBilinearForm a(element);
  Matrix A;
  tic();
  for (unsigned int i = 0; i < N; i++)
    NewFEM::assemble(a, mesh, A);

  return toc();
}

// Test new assembly (FFC)
real testFFC(Mesh& mesh)
{
  cout << "--- Testing new assembly, FFC ---" << endl;

  FFCPoissonFiniteElement element;
  FFCPoissonBilinearForm a(element);
  Matrix A;
  tic();
  for (unsigned int i = 0; i < N; i++)
    NewFEM::assemble(a, mesh, A);

  return toc();
}

// Test new assembly (FFC) with PETSc
real testFFCPETSc(Mesh& mesh)
{
  cout << "--- Testing new assembly, FFC + PETSc ---" << endl;
  
  FFCPoissonFiniteElement element;
  FFCPoissonBilinearForm a(element);
  NewMatrix A;
  tic();
  for (unsigned int i = 0; i < N; i++)
    NewFEM::testPETSc(a, mesh, A);

  return toc();
}

int testAssembly(Mesh& mesh)
{
  dolfin_log(false);
  
  real t1 = testOld(mesh);
  real t2 = testOptimized(mesh);
  real t3 = testFFC(mesh);
  real t4 = testFFCPETSc(mesh);

  dolfin_log(true);

  cout << "---------------------------------------------" << endl;
  cout << "Mesh size: " << mesh.noNodes() << " nodes, and " << mesh.noCells() << " cells" << endl;
  cout << "DOLFIN + DOLFIN: " << t1 << endl;
  cout << "DOLFIN + OPTIM:  " << t2 << endl;
  cout << "DOLFIN + FFC:    " << t3 << endl;
  cout << "PETSc  + FFC:    " << t4 << endl;
  cout << "---------------------------------------------" << endl;

  return 0;
}


int main()
{
  dolfin_set("output", "plain text");
  
  Mesh mesh("mesh.xml.gz");
  testAssembly(mesh);
  for (int i=0; i<M; i++){
    mesh.refineUniformly();
    testAssembly(mesh);
  }
  
  return 0;
}
