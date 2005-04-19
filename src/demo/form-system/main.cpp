// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>
#include <iostream>
//#include "OptimizedPoisson.h"
//#include "FFCPoisson.h"
//#include "Poisson.h" 
//#include "OldPoisson.h" 
//#include "PoissonSystem.h" 
#include "Elasticity.h" 
//#include "OldElasticity.h" 

using namespace dolfin;

#define N 1 // Number of times to do the assembly
#define M 1 // Number of times to refine the mesh 

// // Test old assembly
// real testOldPoisson(Mesh& mesh)
// {
//   cout << "--- Testing old assembly ---" << endl;

//   OldPoisson poisson;
//   Matrix A;
//   tic();
//   for (unsigned int i = 0; i < N; i++)
//     FEM::assemble(poisson, mesh, A);

//   cout << "A (Old Poisson): " << endl;

//   NewMatrix Anew(A);
//   Anew.disp(false);

//   return toc();
// }

// // Test new assembly (FFC)
// real testFFC(Mesh& mesh)
// {
//   NewMatrix B;

//   cout << "--- Testing new assembly, FFC ---" << endl;

//   Poisson::FiniteElement element;
//   Poisson::BilinearForm a;
//   NewMatrix A;
//   tic();

//   for (unsigned int i = 0; i < N; i++)
//     NewFEM::assemble(a, A, mesh, element);

//   cout << "A (FFC Poisson): " << endl;
//   A.disp(false);

//   return toc();
// }

// // Test new assembly (FFC)
// real testFFCSystem(Mesh& mesh)
// {
//   NewMatrix B;

//   cout << "--- Testing new assembly, FFC ---" << endl;

//   PoissonSystem::FiniteElement element;
//   PoissonSystem::BilinearForm a;
//   NewMatrix A;
//   tic();

//   for (unsigned int i = 0; i < N; i++)
//     NewFEM::assemble(a, A, mesh, element);

//   cout << "A (FFC Poisson System): " << endl;
//   A.disp(false);

//   return toc();
// }

// // Test old assembly
// real testOldElasticity(Mesh& mesh)
// {
//   cout << "--- Testing old assembly (Elasticity) ---" << endl;

//   OldElasticity elasticity;
//   Matrix A;
//   tic();
//   for (unsigned int i = 0; i < N; i++)
//     FEM::assemble(elasticity, mesh, A);

//   cout << "A (Old Elasticity): " << endl;
//   NewMatrix Anew(A);
//   Anew.disp(false);

//   NewMatrix Anewp(A.size(0), A.size(1));

//   for(unsigned int i = 0; i < Anewp.size(0); i++)
//   {
//     for(unsigned int j = 0; j < Anewp.size(1); j++)
//     {
//       //cout << "i: " << i << " " << 3 * (i % 4) + (i / 4) << endl;
//       //cout << "j: " << j << " " << 4 * (j % 3) + (j / 3) << endl;
//       Anewp(4 * (i % 3) + (i / 3), 4 * (j % 3) + (j / 3)) = Anew(i, j);
//     }
//   }

//   cout << "A (Old Elasticity permuted): " << endl;
//   Anewp.disp(false);

//   return toc();
// }

// Test new assembly (FFC)
real testElasticity(Mesh& mesh)
{
  Matrix B;

  cout << "--- Testing new assembly, FFC ---" << endl;

  real lambda = 1.0;  // Lame coefficient
  real mu = 1.0;  // Lame coefficient

  //Elasticity::FiniteElement element;
  Elasticity::BilinearForm a(lambda, mu);
  Matrix A;
  tic();

  for (unsigned int i = 0; i < N; i++)
    NewFEM::assemble(a, A, mesh);

  cout << "A (FFC elasticity): " << endl;
  A.disp(false);

  return toc();
}

int testAssembly(Mesh& mesh2D, Mesh& mesh3D)
{
  //dolfin_log(false);
  
  //real t1 = testFFC(mesh2D);
  //real t2 = testOldPoisson(mesh2D);
  //real t3 = testFFCSystem(mesh3D);
  real t4 = testElasticity(mesh3D);
  //real t5 = testOldElasticity(mesh3D);

  //dolfin_log(true);

  cout << "Mesh2D size: " << mesh2D.noNodes() << " nodes, and " << mesh2D.noCells() << " cells" << endl;
  cout << "Mesh3D size: " << mesh3D.noNodes() << " nodes, and " << mesh3D.noCells() << " cells" << endl;
  cout << "---------------------------------------------" << endl;
  //cout << "DOLFIN + FFC:    " << t1 << endl;
  //cout << "DOLFIN + FFC (OldPoisson):    " << t2 << endl;
  //cout << "DOLFIN + FFC (System):    " << t3 << endl;
  cout << "DOLFIN + FFC (Elasticity):    " << t4 << endl;
  //cout << "DOLFIN + FFC (OldElasticity):    " << t5 << endl;
  //cout << "---------------------------------------------" << endl;

  return 0;
}


int main()
{
  //Mesh mesh("mesh.xml.gz");
  Mesh mesh2D("minimal2.xml.gz");
  Mesh mesh3D("minimal.xml.gz");
  testAssembly(mesh2D, mesh3D);

  return 0;
}
