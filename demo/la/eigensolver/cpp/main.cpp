// Copyright (C) 2007 Kristian B. Oelgaard and Garth N. Wells
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
//
// First added:  2007-03-08
// Last changed: 2008-08-28
//
// This simple program illustrates the use of the SLEPc eigenvalue solver.

#include <dolfin.h>
  
using namespace dolfin;

int main()
{
  #ifdef HAS_SLEPC

  // Make sure we use the PETSc backend
  dolfin_set("linear algebra backend", "PETSc");

  // Build stiftness matrix
  UnitSquare mesh(64, 64);
  StiffnessMatrix A(mesh);

  // Get PETSc matrix
  PETScMatrix& AA(A.down_cast<PETScMatrix>());
  cout << AA << endl;

  // Compute the first n eigenvalues
  unsigned int n = 10;
  SLEPcEigenSolver esolver;
  esolver.set("eigenvalue spectrum", "smallest magnitude");
  esolver.solve(AA, n);

  // Display eigenvalues
  for (unsigned int i = 0; i < n; i++)
  {
    real lr, lc;
    esolver.getEigenvalue(lr, lc, i);
    cout << "Eigenvalue " << i << ": " << lr << endl;
  }

  #else

    cout << "SLEPc must be installed to run this demo." << endl;

  #endif

  return 0;

}
