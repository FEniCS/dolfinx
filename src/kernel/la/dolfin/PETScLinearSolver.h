// Copyright (C) 2004-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2004-06-19
// Last changed: 2006-08-08

#ifndef __PETSC_LINEAR_SOLVER_H
#define __PETSC_LINEAR_SOLVER_H

#ifdef HAVE_PETSC_H

#include <dolfin/dolfin_log.h>

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
    virtual uint solve(const PETScMatrix& A, PETScVector& x, const PETScVector& b) = 0;

  };

}

#endif

#endif
