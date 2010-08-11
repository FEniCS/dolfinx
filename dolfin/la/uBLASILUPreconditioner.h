// Copyright (C) 2006-2009 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
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

  template<class Mat> class uBLASMatrix;
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
    std::vector<uint> diagonal;

    const Parameters& krylov_parameters;

  };

}

#endif
