// Copyright (C) 2007 Kristian B. Oelgaard and Garth N. Wells
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-03-08
// Last changed: 2007-03-08
//
// This simple program illustrates the use of the PETScEigenvalueSolver

#include <dolfin.h>
  
using namespace dolfin;

int main()
{
  #ifdef HAVE_SLEPC_H

  // Set up two simple test matrices (2 x 2)
  PETScMatrix A(2,2);
  A(0,0) = 4;
  A(0,1) = 1;
  A(1,0) = 3;
  A(1,1) = 2;

  cout << "Matrix A:" << endl;
  A.disp();

  PETScMatrix B(2,2);
  B(0,0) = 1;
  B(0,1) = 0;
  B(1,0) = 0;
  B(1,1) = 1;

  cout << "Matrix B:" << endl;
  B.disp();

  // Create eigensolver of type LAPACK
  SLEPcEigenvalueSolver esolver(SLEPcEigenvalueSolver::lapack);

  // Compute all eigenpairs of the generalised problem Ax = \lambda Bx
  esolver.solve(A, B);

  // Real and imaginary parts of an eigenvalue  
  real err, ecc;        

  // Real and imaginary parts of an eigenvectora;  
  PETScVector rr(2), cc(2);  

  // Get the first eigenpair from the solver
  const dolfin::uint emode = 0; 
  esolver.getEigenpair(err, ecc, rr, cc, emode);

  // Display result
  cout<< "Eigenvalue, mode: "<< emode << ", real: " << err << ", imag: " << ecc << endl;
  cout<< "Eigenvalue vectors (real and complex parts): "<< endl;
  rr.disp();
  cc.disp();

  #else

    cout << "SLEPc must be installed to run this demo." << endl;

  #endif

  return 0;
}
