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
// Last changed: 2012-12-12

#ifndef __LINEAR_OPERATOR_H
#define __LINEAR_OPERATOR_H

#include <memory>
#include <dolfin/common/mpi.h>
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

    /// Create linear operator
    LinearOperator();

    // Create linear operator to match parallel layout of vectors
    // x and y for product y = Ax.
    LinearOperator(const GenericVector& x, const GenericVector& y);

    /// Destructor
    virtual ~LinearOperator() {}

    /// Return size of given dimension
    virtual std::size_t size(std::size_t dim) const = 0;

    /// Compute matrix-vector product y = Ax
    virtual void mult(const GenericVector& x, GenericVector& y) const = 0;

    /// Return the MPI communicator
    virtual MPI_Comm mpi_comm() const
    { dolfin_assert(_matA); return _matA->mpi_comm(); }

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    //--- Special functions, intended for library use only ---

    /// Return concrete instance / unwrap (const version)
    virtual const GenericLinearOperator* instance() const;

    /// Return concrete instance / unwrap (non-const version)
    virtual GenericLinearOperator* instance();

    /// Return concrete instance / unwrap (const shared pointer version)
    virtual std::shared_ptr<const LinearAlgebraObject> shared_instance() const;

    /// Return concrete instance / unwrap (shared pointer version)
    virtual std::shared_ptr<LinearAlgebraObject> shared_instance();

  private:

    // Pointer to concrete implementation
    std::shared_ptr<GenericLinearOperator> _matA;

  };

}

#endif
