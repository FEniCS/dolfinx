// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __KRYLOV_SOLVER_HH
#define __KRYLOV_SOLVER_HH

#include "SparseMatrix.hh"
#include <dolfin/Vector.h>

namespace dolfin {
  
  enum KrylovMethod {gmres,cg};
  
  enum Preconditioner {pc_richardson,pc_jacobi,pc_gaussseidel,pc_sor,pc_none};
  
  class KrylovSolver{
  public:
	 
	 KrylovSolver();
	 ~KrylovSolver(){}
	 
	 void Solve(SparseMatrix *A, Vector *x, Vector *b);
	 void Solve(Vector *x, Vector *b);
	 
	 void SetMethod(KrylovMethod krylov_method);
	 void SetMatrix(SparseMatrix* A);
	 
  private:
	 
	 void SolveCG    (Vector *xvec, Vector* b);
	 void SolveGMRES(Vector* x, Vector* b);
	 real SolveGMRES_restart_k(Vector* x, Vector* b, int k_max);
	 
	 void SolvePxv   (Vector *x);
	 
	 void ApplyMatrix(Vector *x, Vector *Ax);
	 void ApplyMatrix(real **x, int comp);
	 
	 void ComputeResidual(Vector *x, Vector *b, Vector *res );
	 real GetResidual(Vector *x, Vector *b);
	 
	 bool TestForOrthogonality (real **v );
	 
	 void AllocateArrays (int n, int k_max);
	 void DeleteArrays   (int n, int k_max);
	 
	 KrylovMethod krylov_method;
	 
	 real tol;
	 real norm_residual;
	 real norm_b;
	 
	 int no_iterations;
	 
	 int no_pc_sweeps;
	 
	 Preconditioner pc;
	 
	 int iteration;
	 
	 SparseMatrix *A;
	 
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
