// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __SI_SOLVER_H
#define __SI_SOLVER_H

#include <dolfin/constants.h>

namespace dolfin {

  class Vector;
  class Matrix;

  class SISolver{
  public:

	 enum Method { RICHARDSON, JACOBI, GAUSS_SEIDEL, SOR };
	
	 SISolver();
	 ~SISolver(){}
	 
	 void solve(Matrix& A, Vector& x, Vector& b);
	 
	 void set(Method method);
	 void set(int noit);
	 
  private:
	 
	 void iterateRichardson  (Matrix& A, Vector& x, Vector& b);
	 void iterateJacobi      (Matrix& A, Vector& x, Vector& b);
	 void iterateGaussSeidel (Matrix& A, Vector& x, Vector& b);
	 void iterateSOR         (Matrix& A, Vector& x, Vector& b);
	 
	 void computeResidual(Matrix& A, Vector& x, Vector& b);
	 
	 Method iterative_method;
	 
	 real tol;
	 real residual;
	 
	 int iteration;
	 
	 int max_no_iterations;
  };
  
  typedef SISolver SimpleSolver;

}
  
#endif
