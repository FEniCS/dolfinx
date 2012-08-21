// Copyright (C) 2012 Garth N. Wells
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

#ifndef __GENERIC_LINEAR_OPERATOR_H
#define __GENERIC_LINEAR_OPERATOR_H

#include <dolfin/log/log.h>

namespace dolfin
{

  /// This class defines a common interface for linear operators,
  /// including actual matrices (class _GenericMatrix_) and linear
  /// operators only defined in terms of their action on vectors
  /// (class _GenericKrylovMatrix_).

  class GenericLinearOperator
  {
  public:

    // Destructor
    virtual ~GenericLinearOperator() {}

    /// Matrix-vector product, y = Ax. The y vector must either be zero-sized
    /// or have correct size and parallel layout.
    virtual void mult(const GenericVector& x, GenericVector& y) const = 0;

  };

}

#endif
