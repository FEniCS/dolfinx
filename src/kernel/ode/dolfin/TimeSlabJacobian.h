// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __TIME_SLAB_JACOBIAN_H
#define __TIME_SLAB_JACOBIAN_H

#include <dolfin/VirtualMatrix.h>

namespace dolfin
{
  
  class ODE;
  class NewMethod;
  class NewTimeSlab;
    
  /// This is the base class for Jacobians defined on mono- or
  /// multi-adaptive time slabs.

  class TimeSlabJacobian : public VirtualMatrix
  {
  public:

    /// Constructor
    TimeSlabJacobian(NewTimeSlab& timeslab);

    /// Destructor
    ~TimeSlabJacobian();

    /// Compute product y = Ax
    virtual void mult(const NewVector& x, NewVector& y) const = 0;

    /// Recompute Jacobian
    void update(const NewTimeSlab& timeslab);

  protected:
    
    // The ODE
    ODE& ode;

    // Method, mcG(q) or mdG(q)
    const NewMethod& method;

    // Values of the Jacobian df/du of the right-hand side
    real* Jvalues;

    // Indices for first element of each row for the Jacobian df/du
    uint* Jindices;
    
  };

}

#endif
