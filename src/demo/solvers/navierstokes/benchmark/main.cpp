// Copyright (C) 2002 Johan Hoffman.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

//dolfin_bc mybc(real x, real y, real z, int node, int component);

real f(real x, real y, real z, real t);

// Boundary conditions
void mybc(BoundaryCondition& bc)
{
  bc.set(BoundaryCondition::DIRICHLET, 0.0);
}

// Test new assembly (FFC)
real testFFC(Mesh& mesh)
{
  cout << "--- Testing new assembly, FFC ---" << endl;

  int N = 2;

  /*
  FFCPoissonFiniteElement element;
  FFCPoissonBilinearForm a(element);
  Matrix A;
  */
  tic();
  for (unsigned int i = 0; i < N; i++)
    //NewFEM::assemble(a, mesh, A);

  return toc();
}

int main()
{
  //Mesh mesh("mesh.xml.gz");
  Mesh mesh("tetmesh-4.xml.gz");
  //Mesh mesh("tetmesh_backward_facing_step_32_8_8.xml.gz");

  //dolfin_log(true);

  testFFC(mesh);

  /*
  Problem navierstokes("navierstokes", mesh);
  
  navierstokes.set("source", f);
  navierstokes.set("boundary condition", mybc);
 
  navierstokes.solve();
  */

  return 0;
}

/*
void mybc(real x, real y, real z, int node, int component)
{
  dolfin_bc bc;

  switch ( component ){
  case 0:
    
    if ( x == 0.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    if ( x == 1.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    if ( y == 0.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    if ( y == 1.0 ){
      bc.type = dirichlet;
      bc.val  = 1.0;
    }
    if ( z == 0.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    if ( z == 1.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }

    break;
  case 1:

    if ( x == 0.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    if ( x == 1.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    if ( y == 0.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    if ( y == 1.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    if ( z == 0.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    if ( z == 1.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }

    break;

  case 2:

    if ( x == 0.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    if ( x == 1.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    if ( y == 0.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    if ( y == 1.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    if ( z == 0.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    if ( z == 1.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    
    break;

  case 3:

    if ( x == 0.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }
    if ( x == 1.0 ){
      bc.type = dirichlet;
      bc.val  = 0.0;
    }

    break;

  }
    
  return bc;
}
*/

real f(real x, real y, real z, real t)
{
  //return (32.0/Re) * (y*(1.0-y) + z*(1.0-z)); 
  return 0.0; 
}
