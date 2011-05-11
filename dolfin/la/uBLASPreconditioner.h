// Copyright (C) 2006-2009 Garth N. Wells
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg, 2006-2008.
//
// First added:  2006-06-23
// Last changed: 2009-07-03

#ifndef __UBLAS_PRECONDITIONER_H
#define __UBLAS_PRECONDITIONER_H

#include <dolfin/log/log.h>

namespace dolfin
{

  class uBLASVector;
  class uBLASKrylovMatrix;
  template<class Mat> class uBLASMatrix;

  /// This class specifies the interface for preconditioners for the
  /// uBLAS Krylov solver.

  class uBLASPreconditioner
  {
  public:

    /// Constructor
    uBLASPreconditioner() {};

    /// Destructor
    virtual ~uBLASPreconditioner() {};

    /// Initialise preconditioner (sparse matrix)
    virtual void init(const uBLASMatrix<ublas_sparse_matrix>& P)
      { error("No init(..) function for preconditioner uBLASMatrix<ublas_sparse_matrix>"); }

    /// Initialise preconditioner (dense matrix)
    virtual void init(const uBLASMatrix<ublas_dense_matrix>& P)
      { error("No init(..) function for preconditioner uBLASMatrix<ublas_dense_matrix>"); }

    /// Initialise preconditioner (virtual matrix)
    virtual void init(const uBLASKrylovMatrix& P)
      { error("No init(..) function for preconditioning uBLASKrylovMatrix"); }

    /// Solve linear system (M^-1)Ax = y
    virtual void solve(uBLASVector& x, const uBLASVector& b) const = 0;

  };

}

#endif
