// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
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
    void computeResiduals(MultiAdaptiveTimeSlab& ts);

    // Propagate time steps according to dependencies
    void propagateDependencies();

    // Multi-adaptive time steps (size N)
    real* timesteps;

    // Multi-adaptive residuals (size N)
    real* residuals;

    // Array for storing temporary data during propagation of time steps (size N)
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
