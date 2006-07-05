// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-01-28
// Last changed: 2006-07-05

#ifndef __MONO_ADAPTIVE_JACOBIAN_H
#define __MONO_ADAPTIVE_JACOBIAN_H

#ifdef HAVE_PETSC_H

#include <dolfin/DenseVector.h>
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

    /// Friends
    friend class MonoAdaptiveNewtonSolver;

    // The time slab
    MonoAdaptiveTimeSlab& ts;

    // True if ODE is implicit
    bool implicit;

    // True if M is piecewise constant
    bool piecewise;

    // FIXME: Maybe we can reuse some other vectors?

    // Temporary vectors for storing multiplication
    mutable DenseVector xx;
    mutable DenseVector yy;

  };

}

#endif

#endif
