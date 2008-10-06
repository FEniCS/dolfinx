// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-01-29
// Last changed: 2005-11-11

#ifndef __MONO_ADAPTIVITY_H
#define __MONO_ADAPTIVITY_H

#include <dolfin/common/types.h>
#include "Controller.h"
#include "Adaptivity.h"

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
    double timestep() const;

    /// Update time step
    void update(double k0, double r, const Method& method, double t, bool first);

  private:

    // Time step controller
    Controller controller;

    // Mono-adaptive time step
    double k;

  };

}

#endif
