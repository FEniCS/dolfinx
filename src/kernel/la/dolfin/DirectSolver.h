// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __DIRECT_SOLVER_H
#define __DIRECT_SOLVER_H

namespace dolfin {
  
  class Matrix;
  class Vector;
  
  class DirectSolver{
  public:
    
    DirectSolver(){}
    ~DirectSolver(){}
    
    /// Solve Ax = b (and replace A by its LU factorization)
    void solve(Matrix& A, Vector& x, const Vector& b) const;

    /// Compute inverse (and replace A by its LU factorization)
    void inverse(Matrix& A, Matrix& Ainv) const;

    /// Solve Ax = b with high precision (not replacing A by its LU factorization)
    void hpsolve(const Matrix& A, Vector& x, const Vector& b) const;

    //---

    /// Compute LU factorization (in-place)
    void lu(Matrix& A) const;

    /// Solve A x = b using a computed lu factorization
    void solveLU(const Matrix& LU, Vector& x, const Vector& b) const;

    /// Compute inverse using a computed lu factorization
    void inverseLU(const Matrix& LU, Matrix& Ainv) const;

    /// Solve A x = b with high precision using a computed lu factorization
    void hpsolveLU(const Matrix& LU, const Matrix& A, Vector& x, const Vector& b) const;

  private:
    
    /// Check that the matrix is dense
    void check(const Matrix& A) const;

  };
  
}
  
#endif
