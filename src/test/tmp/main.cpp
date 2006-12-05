// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005
// Last changed: 2006-08-08
//
// This file is used for testing out new features implemented in the
// library, which means that the contents of this file is constantly
// changing. Anything can be thrown into this file at any time. Use
// this for simple tests that are not suitable to be implemented as
// demos in src/demo.
//
// This file has grown quite large lately. Tests should be moved from
// this file to the unit test framework. Benchmarks should be moved
// to src/bench/. (Need to set up a benchmarking framework.)

#include <dolfin.h>
#include <dolfin/Poisson2D.h>
#include "L2Norm.h"

using namespace dolfin;

void testDofMapping()
{
  UnitSquare mesh(1000,1000);
  Poisson2D::BilinearForm a;

  DofMapping dof_map0(mesh, &a.test());
  DofMapping dof_map1(mesh, &a.test(), &a.test());
  
  // Compute system size
  int size0 = dof_map0.size();
  int size1 = dof_map1.size();
  cout << "DofMapping size " << size0 << "  " << size1 << endl;

  int size = FEM::size(mesh, a.test());
  cout << "FEM size " << size << endl;
  
  // Compute number of nonzeroes per row
  int* nzero = new int[size];
  tic();
  dof_map1.numNonZeroesRow(nzero);
  cout << "Time create sparsity " << toc() << endl;

//  for(int i =0; i < size; ++i)
//   cout << "  sparsity " << i << "  " << nzero[i] << endl;

  delete [] nzero;

/*
  // Print matrix to screen to verify computed sparsity
  dolfin_log(false);
  Matrix A;
  FEM::assemble(a, A, mesh);
  dolfin_log(true);
  A.disp();
*/

}

int main(int argc, char* argv[])
{
  testDofMapping();

  return 0;
}
