// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __TIME_STEPPER_H
#define __TIME_STEPPER_H

#include <dolfin/constants.h>

namespace dolfin {

  class ODE;
  class RHS;
  class File;
  class Solution;
  class TimeSlab;
  class Adaptivity;

  /// TimeStepper computes the solution of a given ODE. This is where
  /// the real work takes place (most of it takes place in the time
  /// slab or even in the local elements), whereas the responsibility
  /// of the ODE solver is also to solve the dual problem (using this
  /// class), compute stability factors and compute error estimates.

  class TimeStepper {
  public:

    /// Solve given ODE
    static void solve(ODE& ode);

  private:

    // Prepare for next time slab
    static void shift(Solution& u, RHS& f, Adaptivity& adaptivity, real t);

    // Save solution (when necessary)
    static void save(Solution& u, RHS& f, TimeSlab& timeslab, File& file,
		     real T, unsigned int no_samples);

  };

}

#endif
