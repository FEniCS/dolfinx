// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>
#include <iostream>
//#include "OptimizedPoisson.h"
//#include "FFCPoisson.h"
//#include "Poisson.h" 
//#include "OldPoisson.h" 
//#include "PoissonSystem.h" 
//#include "Elasticity.h" 
//#include "OldElasticity.h" 

using namespace dolfin;

/*

#define N 1 // Number of times to do the assembly
#define M 1 // Number of times to refine the mesh 


// Test old assembly
real testOldPoisson(Mesh& mesh)
{
  cout << "--- Testing old assembly ---" << endl;

  OldPoisson poisson;
  Matrix A;
  tic();
  for (unsigned int i = 0; i < N; i++)
    FEM::assemble(poisson, mesh, A);

  cout << "A (OldPoisson): " << endl;
  A.show();

  return toc();
}

// Test new assembly (FFC)
real testFFC(Mesh& mesh)
{
  NewMatrix B;

  cout << "--- Testing new assembly, FFC ---" << endl;

  PoissonFiniteElement element;
  PoissonBilinearForm a(element);
  NewMatrix A;
  tic();

  for (unsigned int i = 0; i < N; i++)
    NewFEM::assemble(a, mesh, A);

  cout << "A scalar: " << endl;
  A.disp();

  return toc();
}

// Test new assembly (FFC)
real testFFCSystem(Mesh& mesh)
{
  NewMatrix B;

  cout << "--- Testing new assembly, FFC ---" << endl;

  PoissonSystemFiniteElement element;
  PoissonSystemBilinearForm a(element);
  NewMatrix A;
  tic();

  for (unsigned int i = 0; i < N; i++)
    NewFEM::assemble(a, mesh, A);

  cout << "A system: " << endl;
  A.disp();

  return toc();
}

// Test new assembly (FFC)
real testElasticity(Mesh& mesh)
{
  NewMatrix B;

  cout << "--- Testing new assembly, FFC ---" << endl;

  ElasticityFiniteElement element;
  ElasticityBilinearForm a(element);
  NewMatrix A;
  tic();

  for (unsigned int i = 0; i < N; i++)
    NewFEM::assemble(a, mesh, A);

  cout << "A elasticity: " << endl;
  A.disp();

  return toc();
}

// Test old assembly
real testOldElasticity(Mesh& mesh)
{
  cout << "--- Testing old assembly (Elasticity) ---" << endl;

  OldElasticity elasticity;
  Matrix A;
  tic();
  for (unsigned int i = 0; i < N; i++)
    FEM::assemble(elasticity, mesh, A);

  cout << "A (OldElasticity): " << endl;
  A.show();

  return toc();
}


int testAssembly(Mesh& mesh)
{
  //dolfin_log(false);
  
  real t1 = testFFC(mesh);
  real t2 = testOldPoisson(mesh);
  real t3 = testFFCSystem(mesh);
  //real t4 = testElasticity(mesh);
  real t5 = testOldElasticity(mesh);

  //dolfin_log(true);

  cout << "Mesh size: " << mesh.noNodes() << " nodes, and " << mesh.noCells() << " cells" << endl;
  cout << "---------------------------------------------" << endl;
  cout << "DOLFIN + FFC:    " << t1 << endl;
  cout << "DOLFIN + FFC (OldPoisson):    " << t2 << endl;
  cout << "DOLFIN + FFC (System):    " << t3 << endl;
  //cout << "DOLFIN + FFC (Elasticity):    " << t4 << endl;
  cout << "DOLFIN + FFC (OldElasticity):    " << t5 << endl;
  cout << "---------------------------------------------" << endl;

  return 0;
}

*/

int main()
{

  dolfin_set("output", "plain text");

  //cout << "Doing assembly 10 times..." << endl;

  //std::cout << "Doing assembly 10 times..." << std::endl;
  //std::cerr << "Doing assembly 10 times..." << std::endl;
  
  //Mesh mesh("mesh.xml.gz");
  //Mesh mesh("minimal.xml.gz");
  //testAssembly(mesh);
  /*
  for (unsigned int i = 0; i < M; i++)
  {
    mesh.refineUniformly();
    testAssembly(mesh);
  }
  */  

  return 0;
}
