// Copyright (C) 2007 Kristian B. Oelgaard
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-03-08
// Last changed: 2007-03-08
//
// This simple program illustrates the use of the PETScEigenvalueSolver

#include <dolfin.h>
  
using namespace dolfin;

int main()
{
  // Check if DOLFIN was compiled with SLEPc
  #ifdef HAVE_SLEPC_H

  // Set up two simple test matrices (2 x 2)
  Matrix A(2,2);
  A(0,0)=4;
  A(0,1)=1;
  A(1,0)=3;
  A(1,1)=2;

cout << "Matrix A:" << endl;
  A.disp();

  Matrix B(2,2);
  B(0,0)=1;
  B(0,1)=0;
  B(1,0)=0;
  B(1,1)=1;

cout << "Matrix B:" << endl;
  B.disp();

  // Declare eigensolver of type LAPACK
  PETScEigenvalueSolver esolver(PETScEigenvalueSolver::lapack);

  // Compute all eigenpairs of the generalised problem Ax = \lambda Bx
  esolver.solve(A,B);

  // Post process
  int emode = 0; // Mode number (first eigenmode)

  real err, ecc; // real and imaginary part of the eigenvalue
  Vector rr(2);  // real part of the eigenvector;
  Vector cc(2);  // imaginary part of the eigenvector;

  // Get solution from solver
  esolver.getEigenpair(err, ecc, rr, cc, emode);

  // Display solution
  cout<< "Eigenvalue, mode: "<< emode << ", real: " << err << ", imag: " << ecc <<"\n";

  // Display real and imaginary part of the eigenvector
  cout<< "Eigenvalue vector: "<< endl;
  rr.disp();
  cc.disp();

  #endif
  return 0;
}
