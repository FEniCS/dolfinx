// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __DIRECT_SOLVER_H
#define __DIRECT_SOLVER_H

namespace dolfin {
  
  class DenseMatrix;
  class Vector;
  
  class DirectSolver{
  public:
    
    DirectSolver(){}
    ~DirectSolver(){}
    
    // Solve Ax = b (and replace A by its LU factorization)
    void solve(DenseMatrix& A, Vector& x, Vector& b);
    
    /// Compute LU factorization (in-place)
    void lu(DenseMatrix& A);
    
    /// Solve the linear system A x = b using a computed lu factorization
    void lusolve(DenseMatrix& LU, Vector& x, Vector& b);
    
  };
  
}
  
#endif
