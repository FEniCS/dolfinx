// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __TIME_STEPPER_H
#define __TIME_STEPPER_H

#include <dolfin/constants.h>

namespace dolfin {

  class ODE;
  class TimeSlab;
  class TimeSteppingData;
  class RHS;
  class File;

  /// Used by the ODE solver to integrate an ODE over an interval of
  /// given length.
  ///
  /// This is where the real work takes place (well, actually most of
  /// it takes place in the time slab or even in the local elements),
  /// whereas the responsibility of the ODE solver is also to solve
  /// the dual problem (using this class), compute stability factors
  /// and compute error estimates.

  class TimeStepper {
  public:

    /// Solve given ODE on the interval (t0, t1]
    static void solve(ODE& ode, real t0, real t1);

  private:

    // Save solution (when necessary)
    static void save(TimeSlab& timeslab, TimeSteppingData& data, RHS& f, 
		     File& file, real t0, real t1, unsigned int no_samples);

  };

}

#endif
