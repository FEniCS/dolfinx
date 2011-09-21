// Copyright (C) 2006 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2006-06-30
// Last changed: 2009-09-08

#ifndef __UBLAS_KRYLOV_MATRIX_H
#define __UBLAS_KRYLOV_MATRIX_H

#include <dolfin/common/types.h>
#include "ublas.h"

namespace dolfin
{

  class uBLASVector;
  template<typename Mat> class uBLASMatrix;

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

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

  private:

    // Temporary data for LU factorization of a uBLASKrylovMatrix
    uBLASMatrix<ublas_dense_matrix>* AA;
    uBLASVector* ej;
    uBLASVector* Aj;

  };

}

#endif
