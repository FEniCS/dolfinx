// Copyright (C) 2005-2006 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2005-01-29
// Last changed: 2006-04-21

#ifndef __MULTI_ADAPTIVITY_H
#define __MULTI_ADAPTIVITY_H

#include <dolfin/common/types.h>
#include "Controller.h"
#include "Adaptivity.h"

namespace dolfin
{
  class ODE;
  class Method;
  class MultiAdaptiveTimeSlab;

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

    /// Return residual for given component
    real residual(uint i) const;

    /// Update time steps
    void update(MultiAdaptiveTimeSlab& ts, real t, bool first);

  private:

    // Compute maximum residuals for components
    void compute_residuals(MultiAdaptiveTimeSlab& ts);

    // Propagate time steps according to dependencies
    void propagate_dependencies();

    // Multi-adaptive time steps (size N)
    real* timesteps;

    // Multi-adaptive residuals (size N)
    real* residuals;

    // std::vector for storing temporary data during propagation of time steps (size N)
    real* ktmp;

    // Values of right-hand side at quadrature points (size m)
    real* f;

    // Maximum local residual on time slab
    real rmax;

    // Maximum local error on time slab
    real emax;

  };

}

#endif
