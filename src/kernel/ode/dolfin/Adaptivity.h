// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ADAPTIVITY_H
#define __ADAPTIVITY_H

#include <dolfin/NewArray.h>
#include <dolfin/Regulator.h>

namespace dolfin {

  class ODE;
  class TimeSlab;
  class Solution;
  class RHS;

  /// Adaptivity controls the adaptive time-stepping.

  class Adaptivity {
  public:

    /// Constructor
    Adaptivity(ODE& ode);

    /// Destructor
    ~Adaptivity();

    /// Return time step regulator for given component
    Regulator& regulator(unsigned int i);

    /// Return time step regulator for given component
    const Regulator& regulator(unsigned int i) const;

    /// Return tolerance
    real tolerance() const;

    /// Return maximum time step
    real maxstep() const;

    /// Return minimum time step
    real minstep() const;
    
    /// Use a stabilizing time step sequence
    void stabilize(real k, unsigned int m);
    
    /// Return whether we use fixed time steps or not
    bool fixed() const;

    /// Return threshold for reaching end of interval
    real threshold() const;

    /// Return number of components
    unsigned int size() const;

    /// Prepare for next time slab
    void shift(Solution& u, RHS& f);

    /// Check if the time slab can be accepted
    bool accept(TimeSlab& timeslab, RHS& f);

    /// Adjust maximum time step
    void adjustMaximumTimeStep(real kmax);

  private:

    // Regulators, one for each component
    NewArray<Regulator> regulators;

    // Tolerance
    real TOL;

    // Maximum and minimum allowed time step
    real kmax;

    // Current maximum time step
    real kmax_current;

    // Flag for fixed time steps
    bool kfixed;
    
    // Threshold for reaching end of interval
    real beta;

    // Remaining number of small time steps
    unsigned int m;

  };

}

#endif
