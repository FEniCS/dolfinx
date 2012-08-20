// Copyright (C) 2005-2010 Anders Logg and Garth N. Wells
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
// Modified by Andy R. Terrel, 2005.
//
// First added:  2005-01-17
// Last changed: 2010-08-28

#ifndef __PETSC_KRYLOV_MATRIX_H
#define __PETSC_KRYLOV_MATRIX_H

#ifdef HAS_PETSC

#include <string>
#include <dolfin/common/types.h>
#include "PETScBaseMatrix.h"

namespace dolfin
{

  class PETScVector;

  // This is the PETSc version of the _GenericKrylovMatrix_
  // (matrix-free) interface for the solution of linear systems
  // defined in terms of the action (matrix-vector product) of a
  // linear operator.

  class PETScKrylovMatrix : public PETScBaseMatrix
  {
  public:

    /// Constructor
    PETScKrylovMatrix();

    // FIXME: Remove?
    /// Create a virtual matrix matching the given vectors
    PETScKrylovMatrix(uint m, uint n);

    // FIXME: Remove?
    /// Resize virtual matrix
    void resize(uint m, uint n);

    //--- Implementation of the GenericTensor interface ---

    /// Compute matrix-vector product y = Ax
    virtual void mult(const GenericVector& x, GenericVector& y) const = 0;

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const;

  };

}

#endif

#endif
