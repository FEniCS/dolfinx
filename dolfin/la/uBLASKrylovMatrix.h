// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-06-30
// Last changed: 2006-07-03

#ifndef __UBLAS_KRYLOV_MATRIX_H
#define __UBLAS_KRYLOV_MATRIX_H

#include <dolfin/common/types.h>
#include "ublas.h"

namespace dolfin
{

  class uBLASVector;
  template<class Mat> class uBLASMatrix;

  /// This class provides an interface for matrices that define linear
  /// systems for the uBLASKrylovSolver. This interface is implemented
  /// by the classes uBLASSparseMatrix and DenseMatrix. Users may also
  /// overload the mult() function to specify a linear system only in
  /// terms of its action.

  class uBLASKrylovMatrix
  {
  public:

    /// Constructor
    uBLASKrylovMatrix() : AA(0), ej(0), Aj(0) {};

    /// Destructor
    virtual ~uBLASKrylovMatrix() {};

    /// Return number of rows (dim = 0) or columns (dim = 1) 
    virtual uint size(uint dim) const = 0;

    /// Compute product y = Ax
    virtual void mult(const uBLASVector& x, uBLASVector& y) const = 0;

    /// Solve linear system Ax = b for a Krylov matrix using uBLAS and dense matrices
    void solve(uBLASVector& x, const uBLASVector& b);

    /// Display matrix 
//    void disp(const int precision = 2) const;

  private:

    // Temporary data for LU factorization of a uBLASKrylovMatrix
    uBLASMatrix<ublas_dense_matrix>* AA;
    uBLASVector* ej;
    uBLASVector* Aj;

  };

}

#endif
