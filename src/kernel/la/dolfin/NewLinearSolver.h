// Copyright (C) 2004-2007 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2004-06-19
// Last changed: 2006-08-07

#ifndef __NEW_LINEAR_SOLVER_H
#define __NEW_LINEAR_SOLVER_H

namespace dolfin
{

  /// Forward declarations
  class NewMatrix;
  class NewVector;

  /// This class defines the interfaces for default linear solvers for
  /// systems of the form Ax = b.

  class NewLinearSolver
  {
  public:

    /// Constructor
    NewLinearSolver(){}

    /// Destructor
    virtual ~NewLinearSolver(){}

    /// Solve linear system Ax = b
    virtual unsigned int solve(const NewMatrix& A, NewVector& x, const NewVector& b) = 0;

  };

}

#endif
