// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SI_SOLVER_HH
#define __SI_SOLVER_HH

#include "SparseMatrix.hh"
#include <dolfin/Vector.h>

namespace dolfin {
  
  enum Method {richardson,jacobi,gaussseidel,sor};
  
  class SISolver{
  public:
	 
	 SISolver();
	 ~SISolver(){}
	 
	 void Solve(SparseMatrix *A, Vector *x, Vector *b);
	 
	 void SetMethod(Method mtd);
	 void SetNoIterations(int noit);
	 
  private:
	 
	 void IterateRichardson  (SparseMatrix *A, Vector *x, Vector *b);
	 void IterateJacobi      (SparseMatrix *A, Vector *x, Vector *b);
	 void IterateGaussSeidel (SparseMatrix *A, Vector *x, Vector *b);
	 void IterateSOR         (SparseMatrix *A, Vector *x, Vector *b);
	 
	 void ComputeResidual(SparseMatrix *A, Vector *x, Vector *b);
	 
	 Method iterative_method;
	 
	 real tol;
	 real residual;
	 
	 int iteration;
	 
	 int max_no_iterations;
  };
  
  typedef SISolver SimpleSolver;

}
  
#endif
