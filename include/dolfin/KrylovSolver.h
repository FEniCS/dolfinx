// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// Needs to be fixed:
//
//   - Check for reorthogonalisation
//   - Use Vector and DenseMatrix rather than real* real**

#ifndef __KRYLOV_SOLVER_H
#define __KRYLOV_SOLVER_H

#include <dolfin/SparseMatrix.h>
#include <dolfin/Vector.h>

namespace dolfin {
  
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

	 void solvePxv     (Matrix &A, Vector &x);
	 
	 void applyMatrix(Matrix &A, Vector *x, Vector *Ax);
	 void applyMatrix(Matrix &A, real **x, int comp);
	 
	 void residual (Matrix &A, Vector &x, Vector &b, Vector &r );
	 real residual (Matrix &A, Vector &x, Vector &b);
	 
	 bool TestForOrthogonality (real **v );
	 
	 void allocArrays (int n, int k_max);
	 void deleteArrays(int n, int k_max);
	 
	 Method method;
	 Preconditioner pc;
	 
	 real tol;
	 real norm_b;
	 
	 int no_iterations;
	 int no_pc_sweeps;
	 int iteration;
	 	 
	 // Arrays for tempoary data storage
	 real **mat_H;
	 real **mat_r;
	 real **mat_v;
	 
	 real *vec_s;
	 real *vec_y;
	 real *vec_w;
	 real *vec_c;
	 real *vec_g;
	 
  };

}
  
#endif
