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

  class uBlasVector;
  class uBlasKrylovMatrix;
  template<class Mat> class uBlasMatrix;

  /// This class specifies the interface for preconditioners for the
  /// uBlas Krylov solver.

  class uBlasPreconditioner : public Parametrized
  {
  public:

    /// Constructor
    uBlasPreconditioner() {};

    /// Destructor
    virtual ~uBlasPreconditioner() {};

    /// Initialise preconditioner (dense matrix)
    virtual void init(const uBlasMatrix<ublas_dense_matrix>& A) {};

    /// Initialise preconditioner (dense matrix)
    virtual void init(const uBlasMatrix<ublas_sparse_matrix>& A) {};

    /// Initialise preconditioner (virtual matrix)
    virtual void init(const uBlasKrylovMatrix& A) {};

    /// Solve linear system (M^-1)Ax = y
    virtual void solve(uBlasVector& x, const uBlasVector& b) const = 0;

  };

}

#endif
