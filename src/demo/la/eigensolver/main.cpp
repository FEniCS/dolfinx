// Copyright (C) 2007 Kristian B. Oelgaard and Garth N. Wells
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-03-08
// Last changed: 2007-05-16
//
// This simple program illustrates the use of the PETScEigenvalueSolver

#include <dolfin.h>
  
using namespace dolfin;

int main()
{
  #ifdef HAVE_SLEPC_H

  // Set up two simple test matrices (2 x 2)
  real A_array[2][2];
  real B_array[2][2];

  A_array[0][0] = 4;  A_array[0][1] = 1;
  A_array[1][0] = 3;  A_array[1][1] = 2;

  B_array[0][0] = 4;  B_array[0][1] = 0;
  B_array[1][0] = 0;  B_array[1][1] = 1;

  unsigned int position[] = {0, 1};

  PETScMatrix A(2,2);
  A.set(*A_array, 2, position, 2, position);  
  cout << "Matrix A:" << endl;
  A.disp();


  PETScMatrix B(2,2);
  B.set(*B_array, 2, position, 2, position);  
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
