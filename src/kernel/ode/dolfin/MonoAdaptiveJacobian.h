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
    MonoAdaptiveJacobian(MonoAdaptiveTimeSlab& timeslab);

    /// Destructor
    ~MonoAdaptiveJacobian();

    /// Compute product y = Ax
    void mult(const NewVector& x, NewVector& y) const;

  private:

    // Compute product for mcG(q)
    void cGmult(const real x[], real y[]) const;

    // Compute product for mdG(q)
    void dGmult(const real x[], real y[]) const;

    // The time slab
    MonoAdaptiveTimeSlab& ts;

  };

}

#endif
