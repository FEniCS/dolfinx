// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2006-2008.
//
// First added:  2006-06-23
// Last changed: 2008-05-08

#ifndef __UBLAS_PRECONDITIONER_H
#define __UBLAS_PRECONDITIONER_H

#include <dolfin/parameter/Parametrized.h>

namespace dolfin
{

  class uBLASVector;
  class uBLASKrylovMatrix;
  template<class Mat> class uBLASMatrix;

  /// This class specifies the interface for preconditioners for the
  /// uBLAS Krylov solver.

  class uBLASPreconditioner : public Parametrized
  {
  public:

    /// Constructor
    uBLASPreconditioner() {};

    /// Destructor
    virtual ~uBLASPreconditioner() {};

    /// Initialise preconditioner (dense matrix)
    virtual void init(const uBLASMatrix<ublas_dense_matrix>& A) {};

    /// Initialise preconditioner (dense matrix)
    virtual void init(const uBLASMatrix<ublas_sparse_matrix>& A) {};

    /// Initialise preconditioner (virtual matrix)
    virtual void init(const uBLASKrylovMatrix& A) {};

    /// Solve linear system (M^-1)Ax = y
    virtual void solve(uBLASVector& x, const uBLASVector& b) const = 0;

  };

}

#endif
