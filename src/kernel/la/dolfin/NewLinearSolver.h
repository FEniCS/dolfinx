// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_LINEAR_SOLVER_H
#define __NEW_LINEAR_SOLVER_H

#include <dolfin/NewVector.h>
#include <dolfin/NewMatrix.h>
#include <dolfin/VirtualMatrix.h>

namespace dolfin
{

  /// This class defines the interface of all linear solvers for
  /// systems of the form Ax = b.
  
  class NewLinearSolver
  {
  public:

    /// Constructor
    NewLinearSolver();

    /// Destructor
    virtual ~NewLinearSolver();

    /// Solve linear system Ax = b
    virtual void solve(const NewMatrix& A, NewVector& x, const NewVector& b) = 0;
    
    /// Solve linear system Ax = b (matrix-free version)
    virtual void solve(const VirtualMatrix& A, NewVector& x, const NewVector& b) = 0;

  };

}

#endif
