// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-08-31
// Last changed: 

#ifndef __EIGENVALUE_SOLVER_H
#define __EIGENVALUE_SOLVER_H

#include <petscksp.h>
#include <dolfin/Matrix.h>
#include <dolfin/Vector.h>

namespace dolfin
{

  /// This class computes eigenvalues of a matrix. It is 
	/// a wrapper for the eigenvalue solver of PETSc.
  
  class EigenvalueSolver
  {
  public:

    /// Create eigenvalue solver
    EigenvalueSolver();

    /// Destructor
    ~EigenvalueSolver();

    /// Compute all eigenvalues of the matrix A directly
    void eigen(const Matrix& A, Vector& r, Vector& c);
           
  private:

    // PETSc solver pointer
    KSP ksp;

  };

}

#endif
