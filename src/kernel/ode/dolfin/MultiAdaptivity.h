// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-01-29
// Last changed: 2005-11-04

#ifndef __MULTI_ADAPTIVITY_H
#define __MULTI_ADAPTIVITY_H

#include <dolfin/constants.h>
#include <dolfin/Controller.h>
#include <dolfin/Adaptivity.h>

namespace dolfin
{
  
  class ODE;
  class Method;

  /// This class controls the multi-adaptive time-stepping

  class MultiAdaptivity : public Adaptivity
  {
  public:

    /// Constructor
    MultiAdaptivity(const ODE& ode, const Method& method);

    /// Destructor
    ~MultiAdaptivity();

    /// Return time step for given component
    real timestep(uint i) const;

    /// Initialize time step update for system
    void updateInit();

    /// Update time step for given component
    void updateComponent(uint i, real k0, real r, const Method& method);

  private:

    // Multi-adaptive time steps
    real* timesteps;

  };

}

#endif
