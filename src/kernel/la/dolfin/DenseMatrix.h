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
    DenseMatrix(Matrix& A);
    ~DenseMatrix();
    
    /// Initialize to a zero m x n matrix
    void init(int m, int n);
    
    /// Index operator
    real& operator() (int i, int j);

    /// Index operator
    real operator() (int i, int j) const;
    
    /// Return size along dimension dim (0 for row size, 1 for columns size)
    int size(int dim) const;
    
    // Solve the linear system Ax = b (and replace A by its LU factorization)
    void solve(Vector& x, Vector& b);
    
    /// Compute LU factorization (in-place)
    void lu();

    /// Solve the linear system A x = b using a computed lu factorization
    void lusolve(Vector& x, Vector& b);

    friend class DirectSolver;
    
  private:
    
    int m, n;
    real **values;
    int *permutation;
    
  };
  
}

#endif
