// Copyright (C) 2007-2010 Kristian B. Oelgaard and Garth N. Wells
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
// Modified by Marie E. Rognes, 2010.
//
// First added:  2007-03-08
// Last changed: 2010-09-05
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
  Mesh mesh("box_with_dent.xml.gz");

  // Build stiffness matrix
  Matrix A;
  StiffnessMatrix::FunctionSpace V(mesh);
  StiffnessMatrix::BilinearForm a(V, V);
  assemble(A, a);

  // Get PETSc matrix
  PETScMatrix& AA(A.down_cast<PETScMatrix>());

  // Create eigensolver
  SLEPcEigenSolver esolver;

  // Compute all eigenvalues of A x = \lambda x
  esolver.solve(AA);

  // Extract largest (first) eigenpair
  double r, c;
  PETScVector rx(A.size(1));
  PETScVector cx(A.size(1));
  esolver.get_eigenpair(r, c, rx, cx, 0);

  std::cout << "Largest eigenvalue: " << r << std::endl;

  // Initialize function with eigenvector
  Function u(V, rx);

  // Plot eigenfunction
  plot(u);

  #else

    cout << "SLEPc must be installed to run this demo." << endl;

  #endif

  return 0;
}
