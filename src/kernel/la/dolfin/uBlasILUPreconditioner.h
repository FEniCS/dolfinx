// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2006.
//
// First added:  2006-06-23
// Last changed: 2006-07-04

#ifndef __UBLAS_ILU_PRECONDITIONER_H
#define __UBLAS_ILU_PRECONDITIONER_H

#include <dolfin/ublas.h>
#include <dolfin/Array.h>
#include <dolfin/uBlasPreconditioner.h>
#include <dolfin/uBlasMatrix.h>

namespace dolfin
{
  
  template<class Mat> class uBlasMatrix;
  class uBlasVector;

  /// This class implements an incomplete LU factorization (ILU)
  /// preconditioner for the uBlas Krylov solver.

  class uBlasILUPreconditioner : public uBlasPreconditioner
  {
  public:

    /// Constructor
    uBlasILUPreconditioner();

    /// Constructor
    uBlasILUPreconditioner(const uBlasMatrix<ublas_sparse_matrix>& A);

    /// Destructor
    ~uBlasILUPreconditioner();

    /// Solve linear system Ax = b approximately
    void solve(uBlasVector& x, const uBlasVector& b) const;

  private:

    // Initialize preconditioner
    void init(const uBlasMatrix<ublas_sparse_matrix>& A);

    // Preconditioner matrix
    uBlasMatrix<ublas_sparse_matrix> M;

    // Diagonal
    Array<uint> diagonal;

  };

}

#endif
