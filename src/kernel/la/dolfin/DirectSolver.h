// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __DIRECT_SOLVER_H
#define __DIRECT_SOLVER_H

namespace dolfin {
  
  class DenseMatrix;
  class SparseMatrix;
  class Vector;
  
  class DirectSolver{
  public:
	 
	 DirectSolver(){}
	 ~DirectSolver(){}

	 // Solve Ax = b (A will be the LU factorisation)
	 void solve(DenseMatrix &A, Vector &x, Vector &b);

	 // Compute LU factorisation of A
	 void LU(DenseMatrix &A);
	 
	 // Solve Ax = b with given LU factorisation
	 void solveLU(DenseMatrix &LU, Vector &x, Vector &b);
	 
  };
  
}
  
#endif
