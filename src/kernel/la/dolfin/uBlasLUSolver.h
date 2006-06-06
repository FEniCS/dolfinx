// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-31
// Last changed:

#ifndef __UBLAS_LU_SOLVER_H
#define __UBLAS_LU_SOLVER_H

#include <dolfin/Parametrized.h>

#include <dolfin/DenseVector.h>
#include <dolfin/uBlasSparseMatrix.h>
#include <dolfin/LinearSolver.h>


namespace dolfin
{
  /// This class implements the direct solution (LU factorization) for
  /// linear systems of the form Ax = b using uBlas data types.
  
//  class uBlasLUSolver : public LinearSolver, public Parametrized
  class uBlasLUSolver : public LinearSolver, public Parametrized
  {
  public:
    
    /// Constructor
    uBlasLUSolver();

    /// Destructor
    ~uBlasLUSolver();

    /// Solve linear system Ax = b (A can be a dense or sparse uBlas matrix)
    template < class MAT >
    uint solve(const MAT& A, DenseVector& x, const DenseVector& b)
      {
        dolfin_warning("LU solver will be used. This may be slow and consume a lot of memory.");
        
        // FIXME: implement renumbering scheme to speed up LU solve
        A.solve(x, b);
        return 1;
      }

  private:
    
  };

}

#endif
