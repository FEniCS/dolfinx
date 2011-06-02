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

  /// This class represents a matrix-free matrix of dimension m x m.
  /// It is a simple wrapper for a PETSc shell matrix. The interface
  /// is intentionally simple. For advanced usage, access the PETSc
  /// Mat pointer using the function mat() and use the standard PETSc
  /// interface.
  ///
  /// The class PETScKrylovMatrix enables the use of Krylov subspace
  /// methods for linear systems Ax = b, without having to explicitly
  /// store the matrix A. All that is needed is that the user-defined
  /// PETScKrylovMatrix implements multiplication with vectors. Note that
  /// the multiplication operator needs to be defined in terms of
  /// PETSc data structures (Vec), since it will be called from PETSc.

  class PETScKrylovMatrix : public PETScBaseMatrix
  {
  public:

    /// Constructor
    PETScKrylovMatrix();

    /// Create a virtual matrix matching the given vectors
    PETScKrylovMatrix(uint m, uint n);

    /// Destructor
    virtual ~PETScKrylovMatrix();

    /// Resize virtual matrix
    void resize(uint m, uint n);

    /// Compute product y = Ax
    virtual void mult(const PETScVector& x, PETScVector& y) const = 0;

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

  };

}

#endif

#endif
