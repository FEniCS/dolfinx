// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>
#include "FFCPoisson.h"

using namespace dolfin;

#define N 10 // Number of times to do the assembly
#define M 1 // Number of times to refine the mesh 

real testFFCPETSc(Mesh& mesh)
{
  cout << "--- Testing assembly; FFC + PETSc ---" << endl;

  FFCPoissonFiniteElement element;
  FFCPoissonBilinearForm a(element);
  NewMatrix A;
  tic();
  for (unsigned int i = 0; i < N; i++)
    NewFEM::assemble(a, mesh, A);

  return toc();
}

int testAssembly(Mesh& mesh)
{
  dolfin_log(false);
  
  real t1 = testFFCPETSc(mesh);

  dolfin_log(true);

  cout << "Mesh size: " << mesh.noNodes() << " nodes, and " << mesh.noCells() << " cells" << endl;
  cout << "---------------------------------------------" << endl;
  cout << "PETSc  + FFC:    " << t1 << endl;
  cout << "---------------------------------------------" << endl;

  return 0;
}


int main()
{
  dolfin_set("output", "plain text");

  cout << "Doing assembly " << N << " times..." << endl;
  
  Mesh mesh("mesh.xml.gz");
  testAssembly(mesh);
  for (unsigned int i = 0; i < M; i++)
  {
    mesh.refineUniformly();
    testAssembly(mesh);
  }
  
  return 0;
}
