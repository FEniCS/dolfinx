// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-01-29
// Last changed: 2005-11-11

#ifndef __MONO_ADAPTIVITY_H
#define __MONO_ADAPTIVITY_H

#include <dolfin/constants.h>
#include <dolfin/Controller.h>
#include <dolfin/Adaptivity.h>

namespace dolfin
{
  class ODE;
  class Method;

  /// This class controls the mono-adaptive time-stepping

  class MonoAdaptivity : public Adaptivity
  {
  public:

    /// Constructor
    MonoAdaptivity(const ODE& ode, const Method& method);

    /// Destructor
    ~MonoAdaptivity();
    
    /// Return time step
    real timestep() const;

    /// Update time step
    void update(real k0, real r, const Method& method, real t, bool first);

  private:

    // Time step controller
    Controller controller;

    // Mono-adaptive time step
    real k;

  };

}

#endif
