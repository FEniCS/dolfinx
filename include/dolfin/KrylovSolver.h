// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// Needs to be fixed:
//
//   - Check for reorthogonalisation
//   - Use Vector and DenseMatrix rather than real* real**

#ifndef __KRYLOV_SOLVER_H
#define __KRYLOV_SOLVER_H

namespace dolfin {

  class Matrix;
  class Vector;
  
  class KrylovSolver{
  public:

	 enum Method { GMRES, CG };
	 enum Preconditioner { RICHARDSON, JACOBI, GAUSS_SEIDEL, SOR, NONE };
	 
	 KrylovSolver();
	 ~KrylovSolver(){}
	 
	 void solve(Matrix &A, Vector &x, Vector &b);
	 void set(Method method);
	 
  private:
	 
	 void solveCG      (Matrix &A, Vector &x, Vector &b);
	 void solveGMRES   (Matrix &A, Vector &x, Vector &b);
	 real restartGMRES (Matrix &A, Vector &x, Vector &b, int k_max);

	 void residual (Matrix &A, Vector &x, Vector &b, Vector &r);
	 real residual (Matrix &A, Vector &x, Vector &b);
	 
	 Method method;
	 Preconditioner pc;
	 
	 real tol;
	 real norm_b;
	 
	 int no_iterations;
	 int no_pc_sweeps;
	 int iteration;
	 	 
  };

}
  
#endif
