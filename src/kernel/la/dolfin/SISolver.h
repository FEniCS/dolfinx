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

    void setMethod(Method method);
    void setNoSweeps(int max_no_iterations);
    
  private:
    
    void iterateRichardson  (Matrix& A, Vector& x, Vector& b);
    void iterateJacobi      (Matrix& A, Vector& x, Vector& b);
    void iterateGaussSeidel (Matrix& A, Vector& x, Vector& b);
    void iterateSOR         (Matrix& A, Vector& x, Vector& b);
    
    real getResidual(Matrix& A, Vector& x, Vector& b);
    
    Method method;
    
    real tol;
    
    int max_no_iterations;
  };
  
  typedef SISolver SimpleSolver;
}

#endif
