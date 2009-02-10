// Copyright (C) 2003-2009 Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Benjamin Kehlet 2008
//
// First added:  2003
// Last changed: 2009-02-10

#ifndef __TIME_STEPPER_H
#define __TIME_STEPPER_H

#include <dolfin/common/types.h>
#include <dolfin/common/real.h>
#include <dolfin/io/File.h>
#include "ODESolution.h"

namespace dolfin
{

  class ODE;
  class TimeSlab;

  /// TimeStepper computes the solution of a given ODE. This is where
  /// the real work takes place (most of it takes place in the time
  /// slab or even in the local elements), whereas the responsibility
  /// of the class ODESolver is also to solve the dual problem (using
  /// this class), compute stability factors and compute error
  /// estimates.
  ///
  /// This class can be used in two different ways. One way is to call
  /// the solve() function to solve the ODE on the entire time
  /// interval or a part thereof:
  ///
  ///   TimeStepper time_stepper(ode);
  ///   time_stepper.solve(u);          (solve on [0, T])
  ///   time_stepper.solve(u, t0, t1);  (solve on [t0, t1])
  /// 
  /// Alternatively, one may call the step() function repeatedly to
  /// solve the ODE one time slab at a time:
  ///
  ///   TimeStepper time_stepper(ode, u);
  ///   time_stepper.step();
  ///   time_stepper.step();

  class TimeStepper
  {
  public:

    /// Constructor
    TimeStepper(ODE& ode);

    /// Destructor
    ~TimeStepper();

    // Solve ODE on [0, T]
    void solve(ODESolution& u);

    // Solve ODE on [t0, t1]
    void solve(ODESolution& u, real t0, real t1);

    /// Step solution, return current time
    real step(ODESolution& u);

    /// Step solution from t0 to t <= t1, return current time
    real step(ODESolution& u, real t0, real t1);

    /// Set state for ODE
    void set_state(const real* u);

    /// Get state for ODE
    void get_state(real* u);

  private:

    // Save interpolated solution (when necessary)
    void save(ODESolution& u);

    // Save at fixed sample points
    void save_fixed_samples(ODESolution& u);
    
    // Save using adaptive samples
    void save_adaptive_samples(ODESolution& u);

    // Save sample at time t
    void save_sample(ODESolution& u, real t);

    // Check if we have reached end time
    bool at_end(real t, real T) const;

    //--- Time-stepping data ---

    // The ODE being solved
    ODE& ode;

    // The time slab
    TimeSlab* timeslab;

    // Storing the computed solution
    File file;

    // Progress bar
    Progress p;

    // Current time
    real t;

    // True if solution has been stopped
    bool _stopped;

    // True if we should save the solution
    bool save_solution;

    // True if we should use adaptive samples
    bool adaptive_samples;

    // Number of samples to save (for non-adaptive sampling)
    unsigned int num_samples;
    
    // Density of sampling (for adaptive sampling)
    unsigned int sample_density;

  };

}

#endif
