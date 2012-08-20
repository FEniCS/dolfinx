// Copyright (C) 2012 Anders Logg and Garth N. Wells
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
// Last changed: 2012-08-20

#ifndef __GENERIC_KRYLOV_MATRIX_H
#define __GENERIC_KRYLOV_MATRIX_H

namespace dolfin
{

  /// This class defines a common interface for linear operators
  /// defined by their action (matrix-vector multiplication), which is
  /// useful for the definition of matrix-free linear systems.

  class GenericKrylovMatrix : public Variable
  {
  public:

    /// Destructor
    virtual ~GenericKrylovMatrix() {}

    /// Compute matrix-vector product y = Ax
    virtual void mult(const GenericVector& x, GenericVector& y) const = 0;

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const = 0;

  };

}

#endif
