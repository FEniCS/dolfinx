// Copyright (C) 2004-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2004-06-19
// Last changed: 2007-05-15

#ifndef __PETSC_LINEAR_SOLVER_H
#define __PETSC_LINEAR_SOLVER_H

#ifdef HAVE_PETSC_H

namespace dolfin
{

  /// Forward declarations
  class PETScMatrix;
  class PETScVector;

  /// This class defines the interfaces for PETSc linear solvers for
  /// systems of the form Ax = b.

  class PETScLinearSolver
  {
  public:

    /// Constructor
    PETScLinearSolver(){}

    /// Destructor
    virtual ~PETScLinearSolver(){}

    /// Solve linear system Ax = b
    virtual unsigned int solve(const PETScMatrix& A, PETScVector& x, const PETScVector& b) = 0;

  };

}

#endif

#endif
