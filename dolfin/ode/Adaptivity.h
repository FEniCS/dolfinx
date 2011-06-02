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
// First added:  2005-11-04
// Last changed: 2005-11-04

#ifndef __ADAPTIVITY_H
#define __ADAPTIVITY_H

#include <dolfin/common/types.h>
#include "Controller.h"
#include "ODE.h"
namespace dolfin
{

  class ODE;
  class Method;

  /// This is the base class for MonoAdaptivity and MultiAdaptivity,
  /// collecting common functionality for adaptive time-stepping.

  class Adaptivity
  {
  public:

    /// Constructor
    Adaptivity(const ODE& ode, const Method& method);

    /// Destructor
    ~Adaptivity();

    /// Check if current solution can be accepted
    bool accept();

    /// Return threshold for reaching end of interval
    real threshold() const;

    /// Return maximum allowed time step
    real kmax() const;

    /// Friends
    friend class Partition;

  protected:

    // The ODE
    const ODE& ode;

    // The method
    const Method& method;

    // Tolerance
    real tol;

    // Maximum allowed time step
    real _kmax;

    // Threshold for reaching end of interval
    real beta;

    // Safety factor for tolerance
    real safety;

    // Previous safety factor for tolerance
    real safety_old;

    // Maximum allowed safety factor for tolerance
    real safety_max;

    // Total number of rejected time steps
    uint num_rejected;

    // True if we should accept the current solution
    bool _accept;

    // Flag for fixed time steps
    bool kfixed;

  };

}

#endif
