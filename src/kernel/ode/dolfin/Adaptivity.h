// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ADAPTIVITY_H
#define __ADAPTIVITY_H

#include <dolfin/NewArray.h>
#include <dolfin/Regulator.h>

namespace dolfin {

  /// Adaptivity controls the adaptive time-stepping.

  class Adaptivity {
  public:

    /// Constructor
    Adaptivity(unsigned int N);

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

    /// Decrease maximum time step (temporarily) with given factor
    void decreaseTimeStep(real factor);
    
    /// Return whether we use fixed time steps or not
    bool fixed() const;

    /// Return threshold for reaching end of interval
    real threshold() const;

    /// Return number of components
    unsigned int size() const;

    /// Prepare for next time slab
    void shift();

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

  };

}

#endif
