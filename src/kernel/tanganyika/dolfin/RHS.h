// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __RHS_H
#define __RHS_H

#include <dolfin/Vector.h>

#include <dolfin/constants.h>

namespace dolfin {

  class ODE;
  class TimeSlab;
  class TimeSlabData;

  /// RHS takes care of evaluating the right-hand side f(u,t)
  /// for a given component at a given time. The vector u is
  /// updated only for the components which influence the
  /// given component, as determined by the sparsity pattern.

  class RHS {
  public:

    /// Constructor
    RHS(ODE& ode, TimeSlabData& data);

    /// Destructor
    ~RHS();
    
    /// Return current component of f evaluated at time t
    real operator() (int index, int node, real t, TimeSlab* timeslab);
    
  private:

    // Update components that influence the current component at time t
    void update(int index, int node, real t, TimeSlab* timeslab);

    // Solution vector
    Vector u;
    
    // The ODE to be integrated
    ODE* ode;

    // Time slab data
    TimeSlabData* data;

  };   
    
}

#endif
