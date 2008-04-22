// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-06-30
// Last changed: 2006-07-03

#ifndef __UBLAS_KRYLOV_MATRIX_H
#define __UBLAS_KRYLOV_MATRIX_H

#include <dolfin/common/types.h>

namespace dolfin
{

  class uBlasVector;

  /// This class provides an interface for matrices that define linear
  /// systems for the uBlasKrylovSolver. This interface is implemented
  /// by the classes uBlasSparseMatrix and DenseMatrix. Users may also
  /// overload the mult() function to specify a linear system only in
  /// terms of its action.

  class uBlasKrylovMatrix
  {
  public:

    /// Constructor
    uBlasKrylovMatrix() {};

    /// Destructor
    virtual ~uBlasKrylovMatrix() {};

    /// Return number of rows (dim = 0) or columns (dim = 1) 
    virtual uint size(uint dim) const = 0;

    /// Compute product y = Ax
    virtual void mult(const uBlasVector& x, uBlasVector& y) const = 0;

    /// Display matrix 
//    void disp(const int precision = 2) const;

  };

}

#endif
