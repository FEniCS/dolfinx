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
    
    /// Solve Ax = b (and replace A by its LU factorization)
    void solve(DenseMatrix& A, Vector& x, const Vector& b) const;

    /// Compute inverse (and replace A by its LU factorization)
    void inverse(DenseMatrix& A, DenseMatrix& Ainv) const;

    /// Solve Ax = b with high precision (not replacing A by its LU factorization)
    void hpsolve(const DenseMatrix& A, Vector& x, const Vector& b) const;

    //---

    /// Compute LU factorization (in-place)
    void lu(DenseMatrix& A) const;

    /// Solve A x = b using a computed lu factorization
    void solveLU(const DenseMatrix& LU, Vector& x, const Vector& b) const;

    /// Compute inverse using a computed lu factorization
    void inverseLU(const DenseMatrix& LU, DenseMatrix& Ainv) const;

    /// Solve A x = b with high precision using a computed lu factorization
    void hpsolveLU(const DenseMatrix& LU, const DenseMatrix& A, Vector& x, const Vector& b) const;

  };
  
}
  
#endif
