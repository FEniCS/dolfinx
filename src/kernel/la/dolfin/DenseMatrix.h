#ifndef __DENSE_MATRIX_H
#define __DENSE_MATRIX_H

#include <dolfin/Variable.h>
#include <dolfin/constants.h>

namespace dolfin {
  
  class DirectSolver;
  class Matrix;
  class Vector;
  
  /// Dense matrix with given number of rows m and columns n
  class DenseMatrix : public Variable {
  public:
    
    DenseMatrix(int m, int n);
    DenseMatrix(const DenseMatrix& A);
    DenseMatrix(const Matrix& A);
    ~DenseMatrix();
    
    /// Initialize to a zero m x n matrix
    void init(int m, int n);
    
    /// Index operator
    real& operator() (int i, int j);

    /// Index operator
    real operator() (int i, int j) const;
    
    /// Return size along dimension dim (0 for row size, 1 for columns size)
    int size(int dim) const;

    //---
    
    /// Solve Ax = b (and replace A by its LU factorization)
    void solve(Vector& x, const Vector& b);

    /// Compute inverse (and replace A by its LU factorization)
    void inverse(DenseMatrix& Ainv);

    /// Solve Ax = b with high precision (not replacing A by its LU factorization)
    void hpsolve(Vector& x, const Vector& b) const;

    /// Compute LU factorization (in-place)
    void lu();

    /// Solve A x = b using a computed lu factorization
    void solveLU(Vector& x, const Vector& b) const;

    /// Compute inverse using a computed lu factorization
    void inverseLU(DenseMatrix& Ainv) const;

    /// Solve A x = b with high precision using a computed lu factorization
    void hpsolveLU(const DenseMatrix& LU, Vector& x, const Vector& b) const;

    /// Output
    void show() const;
    friend LogStream& operator<< (LogStream& stream, const DenseMatrix& A);

    // Friends
    friend class DirectSolver;
    
  private:
    
    int m, n;
    real **values;
    int *permutation;
    
  };
  
}

#endif
