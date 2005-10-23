// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-10-23
// Last changed: 2005

#ifndef __NONLINEAR_SOLVER_H
#define __NONLINEAR_SOLVER_H

#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/VirtualMatrix.h>

namespace dolfin
{

  /// This class defines the interface of all nonlinear solvers for
  /// equations of the form F(x) = 0.
  
  class NonlinearSolver
  {
  public:

    /// Constructor
    NonlinearSolver();

    /// Destructor
    virtual ~NonlinearSolver();

  };

}

#endif
