// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FIXED_POINT_SOLVER_H
#define __FIXED_POINT_SOLVER_H

#include <dolfin/constants.h>
#include <dolfin/TimeSlabSolver.h>

namespace dolfin
{

  class NewTimeSlab;
  class NewMethod;

  class FixedPointSolver : public TimeSlabSolver
  {
  public:

    /// Constructor
    FixedPointSolver(NewTimeSlab& timeslab, const NewMethod& method);

    /// Destructor
    ~FixedPointSolver();

    /// Solve system
    void solve();

  protected:

    // Make an iteration
    real iteration();

  private:

    real* f; // Values of right-hand side at quadrature points

  };

}

#endif
