// Copyright (C) 2012 Anders Logg
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
// First added:  2012-08-20
// Last changed: 2012-08-21

#ifndef __KRYLOV_MATRIX_H
#define __KRYLOV_MATRIX_H

#include <boost/shared_ptr.hpp>
#include "GenericKrylovMatrix.h"

namespace dolfin
{

  /// This class provides the default DOLFIN Krylov matrix interface
  /// for definition of linear systems based on their action
  /// (matrix-vector multiplication). The linear algebra backend is
  /// decided at run-time based on the present value of the
  /// "linear_algebra_backend" parameter.
  ///
  /// To define a matrix-free matrix, users need to inherit from this
  /// class and overload the function mult(x, y) which defines the
  /// action of the matrix on the vector x as y = Ax.

  class KrylovMatrix : public GenericKrylovMatrix
  {
  public:

    /// Create a Krylov matrix of dimensions M x N
    KrylovMatrix(uint M, uint N);

    /// Destructor
    virtual ~KrylovMatrix() {}

    /// Resize matrix
    void resize(uint M, uint N);

    /// Return size of given dimension
    uint size(uint dim) const;

    /// Compute matrix-vector product y = Ax
    virtual void mult(const GenericVector& x, GenericVector& y) const = 0;

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

  private:

    // Pointer to concrete implementation
    boost::shared_ptr<GenericKrylovMatrix> _A;

  };

}

#endif
