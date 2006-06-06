// Copyright (C) 2004-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2004-06-19
// Last changed: 2006-06-06

#ifndef __LINEAR_SOLVER_H
#define __LINEAR_SOLVER_H

#include <dolfin/dolfin_log.h>

#include <dolfin/Matrix.h>
#include <dolfin/Vector.h>

#include <dolfin/VirtualMatrix.h>


namespace dolfin
{
  /// This class defines the interface for linear solvers for
  /// systems of the form Ax = b.
  
  //FIXME:  These virtual functions have been made non-pure as different solvers
  //        accept different matrix/vector types as arguments. Is there are better
  //        solution? Ideally the uBlas solvers would be templated as they act 
  //        on both dense and sparse data types.
  
  class LinearSolver
  {
  public:

    /// Constructor
    LinearSolver();

    /// Destructor
    virtual ~LinearSolver();

    /// Solve linear system Ax = b
    virtual uint solve(const Matrix& A, Vector& x, const Vector& b) = 0;

#ifdef HAVE_PETSC_H
    /// Solve linear system Ax = b (matrix-free version)
    virtual uint solve(const VirtualMatrix& A, PETScVector& x, const PETScVector& b) = 0;
#endif

  };

}

#endif
