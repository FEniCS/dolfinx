// Copyright (C) 2004-2007 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2004-06-19
// Last changed: 2006-08-07

#ifndef __LINEAR_SOLVER_H
#define __LINEAR_SOLVER_H

#include <dolfin/Matrix.h>
#include <dolfin/Vector.h>

namespace dolfin
{

  /// Forward declarations
  //class Matrix;
  //class Vector;

  /// This class defines the interfaces for default linear solvers for
  /// systems of the form Ax = b.

  class LinearSolver
  {
  public:

    /// Constructor
    LinearSolver(){}

    /// Destructor
    virtual ~LinearSolver(){}

    /// Solve linear system Ax = b
    virtual unsigned int solve(const Matrix& A, Vector& x, const Vector& b) = 0;

  };

}

#endif
