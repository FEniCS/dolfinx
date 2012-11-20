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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg 2006.
//
// First added:  2006-06-23
// Last changed: 2009-07-03

#ifndef __UBLAS_ILU_PRECONDITIONER_H
#define __UBLAS_ILU_PRECONDITIONER_H

#include "ublas.h"
#include "uBLASPreconditioner.h"
#include "uBLASMatrix.h"

namespace dolfin
{

  template<typename Mat> class uBLASMatrix;
  class uBLASVector;

  /// This class implements an incomplete LU factorization (ILU)
  /// preconditioner for the uBLAS Krylov solver.

  class uBLASILUPreconditioner : public uBLASPreconditioner
  {
  public:

    /// Constructor
    uBLASILUPreconditioner(const Parameters& krylov_parameters);

    /// Destructor
    ~uBLASILUPreconditioner();

    // Initialize preconditioner
    void init(const uBLASMatrix<ublas_sparse_matrix>& P);

    /// Solve linear system Ax = b approximately
    void solve(uBLASVector& x, const uBLASVector& b) const;

  private:

    // Preconditioner matrix (factorised)
    uBLASMatrix<ublas_sparse_matrix> M;

    // Diagonal
    std::vector<unsigned int> diagonal;

    const Parameters& krylov_parameters;

  };

}

#endif
