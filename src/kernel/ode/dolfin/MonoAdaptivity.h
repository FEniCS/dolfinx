// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-01-29
// Last changed: 2005-11-04

#ifndef __MONO_ADAPTIVITY_H
#define __MONO_ADAPTIVITY_H

#include <dolfin/constants.h>
#include <dolfin/Controller.h>

// FIXME: Use common base class Adaptivity

namespace dolfin
{
  
  class ODE;
  class Method;

  /// This class controls the mono-adaptive time-stepping

  class MonoAdaptivity
  {
  public:

    /// Constructor
    MonoAdaptivity(ODE& ode, const Method& method);

    /// Destructor
    ~MonoAdaptivity();

    /// Return time step
    real timestep() const;

    /// Update time step
    void update(real k0, real r, const Method& method);

    /// Check if current solution can be accepted
    bool accept();

    /// Return threshold for reaching end of interval
    real threshold() const;

    /// Use a stabilizing time step sequence
    //     void stabilize(real k, uint m);

  private:

    // Mono-adaptive time step
    real k;

    // Time step controller
    Controller controller;

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

    // Safety factor for tolerance
    real safety;

    // Previous safety factor for tolerance
    real safety_old;

    // Maximum allowed safety factor for tolerance
    real safety_max;

    // True if we should accept the current solution
    bool _accept;

    // Number of rejected time steps
    uint num_rejected;

  };

}

#endif
