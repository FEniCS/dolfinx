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
// Last changed: 2012-12-12

#ifndef __GENERIC_LINEAR_OPERATOR_H
#define __GENERIC_LINEAR_OPERATOR_H

#include "LinearAlgebraObject.h"

namespace dolfin
{

  // Forward declarations
  class GenericVector;
  class GenericLinearAlgebraFactory;

  /// This class defines a common interface for linear operators,
  /// including actual matrices (class _GenericMatrix_) and linear
  /// operators only defined in terms of their action on vectors.
  ///
  /// This class is used internally by DOLFIN to define a class
  /// hierarchy of backend independent linear operators and solvers.
  /// Users should not interface to this class directly but instead
  /// use the _LinearOperator_ class.

  class GenericLinearOperator : public virtual LinearAlgebraObject
  {
  public:

    // Destructor
    virtual ~GenericLinearOperator() {}

    /// Return size of given dimension
    virtual std::size_t size(std::size_t dim) const = 0;

    /// Compute matrix-vector product y = Ax
    virtual void mult(const GenericVector& x, GenericVector& y) const = 0;

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const = 0;

  protected:

    friend class LinearOperator;

    // Initialize linear operator to match parallel layout of vectors
    // x and y for product y = Ax. Needs to be implemented by backend.
    virtual void init_layout(const GenericVector& x,
			     const GenericVector& y,
			     GenericLinearOperator* wrapper)
    {
      dolfin_error("GenericLinearOperator.h",
                   "initialize backend implementation of linear operator",
                   "Missing init_layout() function for backend");
    }

  };

}

#endif
