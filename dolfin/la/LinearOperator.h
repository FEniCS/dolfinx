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
// // You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2012-08-20
// Last changed: 2012-08-21

#ifndef __LINEAR_OPERATOR_H
#define __LINEAR_OPERATOR_H

#include <boost/shared_ptr.hpp>
#include "GenericLinearOperator.h"

namespace dolfin
{

  /// This class defines an interface for linear operators defined
  /// only in terms of their action (matrix-vector product) and can be
  /// used for matrix-free solution of linear systems. The linear
  /// algebra backend is decided at run-time based on the present
  /// value of the "linear_algebra_backend" parameter.
  ///
  /// To define a linear operator, users need to inherit from this
  /// class and overload the function mult(x, y) which defines the
  /// action of the matrix on the vector x as y = Ax.

  class LinearOperator : public GenericLinearOperator
  {
  public:

    /// Create a Krylov matrix of dimensions M x N
    LinearOperator(uint M, uint N);

    /// Destructor
    virtual ~LinearOperator() {}

    /// Return size of given dimension
    uint size(uint dim) const;

    /// Compute matrix-vector product y = Ax
    virtual void mult(const GenericVector& x, GenericVector& y) const = 0;

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

  private:

    // Pointer to concrete implementation
    boost::shared_ptr<GenericLinearOperator> _A;

  };

}

#endif
