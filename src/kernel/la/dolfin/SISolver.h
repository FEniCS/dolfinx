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
    
    void solve(const Matrix& A, Vector& x, const Vector& b);

    void setMethod(Method method);
    void setNoSweeps(int max_no_iterations);
    
  private:
    
    void iterateRichardson  (const Matrix& A, Vector& x, const Vector& b);
    void iterateJacobi      (const Matrix& A, Vector& x, const Vector& b);
    void iterateGaussSeidel (const Matrix& A, Vector& x, const Vector& b);
    void iterateSOR         (const Matrix& A, Vector& x, const Vector& b);
    
    real residual(const Matrix& A, Vector& x, const Vector& b);
    
    Method method;
    
    real tol;
    
    int max_no_iterations;
  };
  
  typedef SISolver SimpleSolver;
}

#endif
