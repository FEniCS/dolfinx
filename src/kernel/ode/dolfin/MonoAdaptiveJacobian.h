// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __MONO_ADAPTIVE_JACOBIAN_H
#define __MONO_ADAPTIVE_JACOBIAN_H

#include <dolfin/TimeSlabJacobian.h>

namespace dolfin
{
  
  class MonoAdaptiveTimeSlab;
    
  /// This class represents the Jacobian matrix of the system of
  /// equations defined on a mono-adaptive time slab.

  class MonoAdaptiveJacobian : public TimeSlabJacobian
  {
  public:

    /// Constructor
    MonoAdaptiveJacobian(MonoAdaptiveTimeSlab& timeslab,
			 bool implicit, bool piecewise);

    /// Destructor
    ~MonoAdaptiveJacobian();

    /// Compute product y = Ax
    void mult(const Vector& x, Vector& y) const;

  private:

    // Compute product for mcG(q)
    void cGmult(const real x[], real y[]) const;

    // Compute product for mdG(q)
    void dGmult(const real x[], real y[]) const;

    // The time slab
    MonoAdaptiveTimeSlab& ts;

    // True if ODE is implicit
    bool implicit;

    // True if M is piecewise constant
    bool piecewise;

  };

}

#endif
