// Copyright (C) 2005 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2005-01-29
// Last changed: 2005-11-11

#ifndef __MONO_ADAPTIVITY_H
#define __MONO_ADAPTIVITY_H

#include <dolfin/common/types.h>
#include <dolfin/common/real.h>
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
