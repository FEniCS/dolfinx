// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Jansson 2003, 2004.

#ifndef __TIME_STEPPER_H
#define __TIME_STEPPER_H

#include <dolfin/constants.h>
#include <dolfin/Partition.h>
#include <dolfin/Adaptivity.h>
#include <dolfin/RHS.h>
#include <dolfin/Solution.h>
#include <dolfin/File.h>
#include <dolfin/FixedPointIteration.h>

namespace dolfin {

  class ODE;
  class TimeSlab;

  /// TimeStepper computes the solution of a given ODE. This is where
  /// the real work takes place (most of it takes place in the time
  /// slab or even in the local elements), whereas the responsibility
  /// of the ODE solver is also to solve the dual problem (using this
  /// class), compute stability factors and compute error estimates.
  ///
  /// This class can be used in two different ways. One way is to
  /// call the static method solve() to solve a given ODE:
  ///
  ///   TimeStepper::solve(ode, u);
  ///
  /// Alternatively, one can create a TimeStepper object and use this
  /// for repeatedly time stepping the ODE, one time slab at a time:
  ///
  ///   TimeStepper timeStepper(ode, u);
  ///   timeStepper.step();
  ///   timeStepper.step();

  class TimeStepper {
  public:

    /// Constructor
    TimeStepper(ODE& ode, Function& function);

    /// Destructor
    ~TimeStepper();
    
    /// Solve given ODE
    static void solve(ODE& ode, Function& function);

    /// Step solution, return current time
    real step();

    /// Check if we have reached the end time
    bool finished() const;

  private:

    // Create the first time slab
    bool createFirstTimeSlab();

    // Create a standard (recursive) time slab
    bool createGeneralTimeSlab();

    // Prepare for next time slab
    void shift();

    // Save interpolated solution (when necessary)
    void save(TimeSlab& timeslab);

    // Save at fixed sample points
    void saveFixedSamples(TimeSlab& timeslab);
    
    // Save using adaptive samples
    void saveAdaptiveSamples(TimeSlab& timeslab);

    // Stabilize using a sequence of small time steps
    void stabilize(real K);
    
    //--- Time-stepping data ---

    // Size of system
    unsigned int N;

    // Current time
    real t;

    // End time of computation
    real T;

    // Partition of components into small and large time steps
    Partition partition;

    // Adaptivity, including regulation of the time step
    Adaptivity adaptivity;

    // The solution being computed
    Solution u;

    /// The ODE being solved
    ODE& ode;

    // The right-hand side
    RHS f;

    // Damped fixed point iteration
    FixedPointIteration fixpoint;

    // Storing the computed solution
    File file;
    
    // Progress bar
    Progress p;

    // True if we have reached the given end time
    bool _finished;

    // True if we should save the solution
    bool save_solution;

    // True if we should solve the dual
    bool solve_dual;

    // True if we should use adaptive samples
    bool adaptive_samples;

    // Number of samples to save (for non-adaptive sampling)
    unsigned int no_samples;
    
    // Density of sampling (for adaptive sampling)
    unsigned int sample_density;

  };

}

#endif
