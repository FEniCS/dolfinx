// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __MULTI_ADAPTIVE_JACOBIAN_H
#define __MULTI_ADAPTIVE_JACOBIAN_H

#include <dolfin/TimeSlabJacobian.h>

namespace dolfin
{
  
  class MultiAdaptiveTimeSlab;
    
  /// This class represents the Jacobian matrix of the system of
  /// equations defined on a multi-adaptive time slab.

  class MultiAdaptiveJacobian : public TimeSlabJacobian
  {
  public:

    /// Constructor
    MultiAdaptiveJacobian(MultiAdaptiveTimeSlab& timeslab);

    /// Destructor
    ~MultiAdaptiveJacobian();

    /// Compute product y = Ax
    void mult(const Vector& x, Vector& y) const;

    /// Recompute Jacobian if necessary
    void update();

    /// Friends
    friend class MultiAdaptivePreconditioner;

  private:

    // Compute product for mcG(q)
    void cGmult(const real x[], real y[]) const;

    // Compute product for mdG(q)
    void dGmult(const real x[], real y[]) const;

    // The time slab
    MultiAdaptiveTimeSlab& ts;

    // Values of the Jacobian df/du of the right-hand side
    real* Jvalues;

    // Indices for first element of each row for the Jacobian df/du
    uint* Jindices;
    
    // Lookup table for dependencies to components with smaller time steps
    real* Jlookup;
    
  };

}

#endif
