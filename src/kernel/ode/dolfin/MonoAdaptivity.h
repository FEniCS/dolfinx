// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __MONO_ADAPTIVITY_H
#define __MONO_ADAPTIVITY_H

#include <dolfin/constants.h>
#include <dolfin/Regulator.h>

namespace dolfin
{
  
  class ODE;
  class NewMethod;

  /// This class controls the mono-adaptive time-stepping

  class MonoAdaptivity
  {
  public:

    /// Constructor
    MonoAdaptivity(ODE& ode);

    /// Destructor
    ~MonoAdaptivity();

    /// Return time step
    real timestep() const;

    /// Update time step
    void update(real k0, real r, const NewMethod& method);

    /// Return threshold for reaching end of interval
    real threshold() const;

    /// Use a stabilizing time step sequence
    void stabilize(real k, uint m);

  private:

    // Mono-adaptive time step
    real k;

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

    // Time step conservation
    real w;

  };

}

#endif
