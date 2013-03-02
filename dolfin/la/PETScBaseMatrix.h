// Copyright (C) 2005-2006 Anders Logg
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
// Modified by Fredrik Valdmanis, 2011
//
// First added:  2005-01-17
// Last changed: 2011-09-07

#ifndef __PETSC_BASE_MATRIX_H
#define __PETSC_BASE_MATRIX_H

#ifdef HAS_PETSC

#include <string>
#include <utility>
#include <boost/shared_ptr.hpp>
#include <petscmat.h>

#include <dolfin/common/types.h>
#include <dolfin/common/Variable.h>
#include "PETScObject.h"

namespace dolfin
{

  class PETScMatrixDeleter
  {
  public:
    void operator() (Mat* A)
    {
      if (*A)
        MatDestroy(A);
      delete A;
    }
  };

  class GenericVector;

  /// This class is a base class for matrices that can be used in
  /// PETScKrylovSolver.

  class PETScBaseMatrix : public PETScObject, public virtual Variable
  {
  public:

    /// Constructor
    PETScBaseMatrix() {}

    /// Constructor
    PETScBaseMatrix(boost::shared_ptr<Mat> A) : _A(A) {}

    /// Return number of rows (dim = 0) or columns (dim = 1)
    std::size_t size(std::size_t dim) const;

    /// Return local range along dimension dim
    std::pair<std::size_t, std::size_t> local_range(std::size_t dim) const;

    /// Resize matrix to be compatible with the matrix-vector product
    /// y = Ax. In the parallel case, both size and layout are
    /// important.
    ///
    /// *Arguments*
    ///     dim (std::size_t)
    ///         The dimension (axis): dim = 0 --> z = y, dim = 1 --> z = x
    void resize(GenericVector& z, std::size_t dim) const;

    /// Return PETSc Mat pointer
    boost::shared_ptr<Mat> mat() const
    { return _A; }

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const = 0;

  protected:

    // PETSc Mat pointer
    boost::shared_ptr<Mat> _A;

  };

}

#endif

#endif
