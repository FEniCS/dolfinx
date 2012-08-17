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
// First added:  2012-08-17
// Last changed: 2012-08-17

#ifndef __TENSOR_PRODUCT_MATRIX_H
#define __TENSOR_PRODUCT_MATRIX_H

#include <dolfin/common/types.h>

namespace dolfin
{

  // Forward declarations
  class TensorProductVector;


  /// A _TensorProductMatrix_ is a matrix expressed as a tensor
  /// product (outer product) of a list of matrices:
  ///
  ///   A_ijklmn... = B_ij C_kl D_mn ...
  ///
  /// The tensor rank is twice the number of factors in the tensor
  /// product. A _TensorProductMatrix_ can be multiplied with a
  /// _TensorProductVector_. The matrix-vector multiplication is
  /// defined by
  ///
  ///   (A a)_ikm = A_ijklmn a_jln
  ///
  /// where a is a _TensorProductVector_ with elements a_jln.

  // FIXME: Either inherit from PETScKrylovMatrix or a common interface

  class TensorProductMatrix
  {
  public:

    /// Create tensor product matrix with given number of factors
    TensorProductMatrix(uint num_factors);

    /// Destructor
    virtual ~TensorProductMatrix() {}

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Compute matrix-vector product y = Ax
    void mult(const TensorProductVector& x, TensorProductVector& y) const;

  };

}

#endif
