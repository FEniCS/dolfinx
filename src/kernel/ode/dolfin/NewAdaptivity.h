// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_ADAPTIVITY_H
#define __NEW_ADAPTIVITY_H

#include <dolfin/constants.h>
#include <dolfin/NewArray.h>
#include <dolfin/Regulator.h>

namespace dolfin
{
  
  class ODE;
  class NewMethod;

  /// Adaptivity controls the multi-adaptive time-stepping

  class NewAdaptivity
  {
  public:

    /// Constructor
    NewAdaptivity(ODE& ode);

    /// Destructor
    ~NewAdaptivity();

    /// Return time step for given component
    real timestep(uint i) const;

    /// Update time step
    void update(uint i, real r, const NewMethod& method);

    /// Return threshold for reaching end of interval
    real threshold() const;

    /// Use a stabilizing time step sequence
    void stabilize(real k, uint m);

  private:

    // Regulators, one for each component
    NewArray<Regulator> regulators;
    
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

    // Remaining number of small time steps
    uint m;

  };

}

#endif
