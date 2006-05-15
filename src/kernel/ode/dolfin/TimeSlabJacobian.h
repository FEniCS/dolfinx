// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-01-28
// Last changed: 2006-05-07

#ifndef __TIME_SLAB_JACOBIAN_H
#define __TIME_SLAB_JACOBIAN_H

#ifdef HAVE_PETSC_H

#include <dolfin/Vector.h>
#include <dolfin/VirtualMatrix.h>

namespace dolfin
{
  
  class ODE;
  class Method;
  class TimeSlab;
    
  /// This is the base class for Jacobians defined on mono- or
  /// multi-adaptive time slabs.

  class TimeSlabJacobian : public VirtualMatrix
  {
  public:

    /// Constructor
    TimeSlabJacobian(TimeSlab& timeslab);

    /// Destructor
    ~TimeSlabJacobian();

    /// Compute product y = Ax
    virtual void mult(const Vector& x, Vector& y) const = 0;

    /// Recompute Jacobian if necessary
    virtual void update();

  protected:
    
    // The ODE
    ODE& ode;

    // Method, mcG(q) or mdG(q)
    const Method& method;

  };

}

#endif

#endif
