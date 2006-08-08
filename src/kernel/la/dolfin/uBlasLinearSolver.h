// Copyright (C) 2004-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2004-06-19
// Last changed: 2006-08-08

#ifndef __UBLAS_LINEAR_SOLVER_H
#define __UBLAS_LINEAR_SOLVER_H

#include <dolfin/dolfin_log.h>
#include <dolfin/ublas.h>

namespace dolfin
{

  /// Forward declarations
  class uBlasVector;  
  template<class Mat> class uBlasMatrix;

  /// This class defines the interfaces for uBlas-based linear solvers for
  /// systems of the form Ax = b.

  class uBlasLinearSolver
  {
  public:

    /// Constructor
    uBlasLinearSolver() {}

    /// Destructor
    virtual ~uBlasLinearSolver() {}

    /// Solve linear system Ax = b (A is dense)
    virtual uint solve(const uBlasMatrix<ublas_dense_matrix>& A, uBlasVector& x, 
        const uBlasVector& b) = 0;

    /// Solve linear system Ax = b (A is sparse)
    virtual uint solve(const uBlasMatrix<ublas_sparse_matrix>& A, uBlasVector& x, 
        const uBlasVector& b) = 0;

  };

}

#endif
