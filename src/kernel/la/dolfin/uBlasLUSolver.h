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


namespace dolfin
{
  /// This class implements the direct solution (LU factorization) for
  /// linear systems of the form Ax = b using uBlas data types.
  
//  class uBlasLUSolver : public LinearSolver, public Parametrized
  class uBlasLUSolver : public Parametrized
  {
  public:
    
    /// Constructor
    uBlasLUSolver() {}

    /// Destructor
    ~uBlasLUSolver(){}

    /// Solve linear system Ax = b (A can be a dense or sparse uBlas matrix)
    template < class MAT >
    uint solve(const MAT& A, DenseVector& x, const DenseVector& b)
      {
        // Get parameters
        const bool report = get("LU report");

        if ( report )
        dolfin_info("Solving linear system of size %d x %d (uBlas LU solver).",
		    A.size(0), A.size(1));

        // Solve
        A.solve(x, b);

        return 1;
      }

  private:
    
  };

}

#endif
