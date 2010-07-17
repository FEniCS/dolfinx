// Copyright (C) 2007-2010 Kristian B. Oelgaard and Garth N. Wells
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
//
// First added:  2007-03-08
// Last changed: 2010-01-02
//
// This simple program illustrates the use of the SLEPc eigenvalue solver.

#include <dolfin.h>
#include "StiffnessMatrix.h"

using namespace dolfin;

int main()
{
  #ifdef HAS_SLEPC

  // Make sure we use the PETSc backend
  parameters["linear_algebra_backend"] = "PETSc";

  // Create mesh
  UnitSquare mesh(64, 64);

  // Build stiftness matrix
  Matrix A;
  StiffnessMatrix::FunctionSpace V(mesh);
  StiffnessMatrix::BilinearForm a(V, V);
  assemble(A, a); 

  // Get PETSc matrix
  PETScMatrix& AA(A.down_cast<PETScMatrix>());
  cout << AA << endl;

  // Compute the first n eigenvalues
  unsigned int n = 10;
  SLEPcEigenSolver esolver;
  esolver.parameters["spectrum"] = "largest magnitude";
  esolver.solve(AA, n);

  cout << "Solver converted in " << esolver.get_iteration_number() << " iterations" << endl;

  // Display eigenvalues
  for (unsigned int i = 0; i < n; i++)
  {
    double lr, lc;
    esolver.get_eigenvalue(lr, lc, i);
    cout << "Eigenvalue " << i << ": " << lr << endl;
  }

  #else

    cout << "SLEPc must be installed to run this demo." << endl;

  #endif

  return 0;
}
