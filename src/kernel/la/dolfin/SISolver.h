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
    
    enum SI_method { richardson, jacobi, gauss_seidel, sor };
    
    SISolver();
    ~SISolver(){}
    
    void solve(Matrix& A, Vector& x, Vector& b);
    
    void setMethod(SI_method method);
    void setNoSweeps(int max_no_iterations);
    
  private:
    
    void iterateRichardson  (Matrix& A, Vector& x, Vector& b);
    void iterateJacobi      (Matrix& A, Vector& x, Vector& b);
    void iterateGaussSeidel (Matrix& A, Vector& x, Vector& b);
    void iterateSOR         (Matrix& A, Vector& x, Vector& b);
    
    real getResidual(Matrix& A, Vector& x, Vector& b);
    
    SI_method method;
    
    real tol;
    
    int max_no_iterations;
  };
  
  typedef SISolver SimpleSolver;
}

#endif
