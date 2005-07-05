// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-01-29
// Last changed: 2005

#ifndef __MULTI_ADAPTIVITY_H
#define __MULTI_ADAPTIVITY_H

#include <dolfin/constants.h>
#include <dolfin/Array.h>
#include <dolfin/Regulator.h>

namespace dolfin
{
  
  class ODE;
  class Method;

  /// This class controls the multi-adaptive time-stepping

  class MultiAdaptivity
  {
  public:

    /// Constructor
    MultiAdaptivity(ODE& ode);

    /// Destructor
    ~MultiAdaptivity();

    /// Return time step for given component
    real timestep(uint i) const;

    /// Update time step
    void update(uint i, real k0, real r, const Method& method);

    /// Return threshold for reaching end of interval
    real threshold() const;

    /// Use a stabilizing time step sequence
    void stabilize(real k, uint m);

  private:

    // Regulators, one for each component
    Array<Regulator> regulators;

    // Multi-adaptive time steps
    real* timesteps;

    // Time step regulator
    Regulator regulator;
    
    // Tolerance
    real tol;

    // Maximum allowed time step
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
