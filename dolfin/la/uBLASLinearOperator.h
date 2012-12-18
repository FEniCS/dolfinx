// Copyright (C) 2006-2012 Anders Logg
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
// Last changed: 2012-12-12

#ifndef __UBLAS_LINEAR_OPERATOR_H
#define __UBLAS_LINEAR_OPERATOR_H

#include <string>
#include <dolfin/common/types.h>
#include "GenericLinearOperator.h"

namespace dolfin
{

  class uBLASVector;

  // This is the uBLAS version of the _GenericLinearOperator_
  // (matrix-free) interface for the solution of linear systems
  // defined in terms of the action (matrix-vector product) of a
  // linear operator.

  class uBLASLinearOperator : public GenericLinearOperator
  {
  public:

    /// Constructor
    uBLASLinearOperator();

    //--- Implementation of the GenericLinearOperator interface ---

    /// Return size of given dimension
    virtual std::size_t size(std::size_t dim) const;

    /// Compute matrix-vector product y = Ax
    virtual void mult(const GenericVector& x, GenericVector& y) const;

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const;

  protected:

    // Initialization
    void init(const GenericVector& x, const GenericVector& y,
              GenericLinearOperator* wrapper);

    // Pointer to wrapper
    GenericLinearOperator* _wrapper;

    // Dimensions
    std::size_t M, N;

  };

}

#endif
