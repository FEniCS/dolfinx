// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEWTON_SOLVER_H
#define __NEWTON_SOLVER_H

#include <dolfin/constants.h>
#include <dolfin/TimeSlabSolver.h>

namespace dolfin
{

  class NewTimeSlab;
  class NewMethod;

  class NewtonSolver : public TimeSlabSolver
  {
  public:

    /// Constructor
    NewtonSolver(NewTimeSlab& timeslab, const NewMethod& method);

    /// Destructor
    ~NewtonSolver();

    /// Solve system
    void solve();

  protected:

    // Iteration
    real iteration();

  private:

    real* f; // Values of right-hand side at quadrature points

  };

}

#endif
