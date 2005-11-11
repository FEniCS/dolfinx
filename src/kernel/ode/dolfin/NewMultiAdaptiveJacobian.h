// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-01-27
// Last changed: 2005-11-10

#ifndef __NEW_MULTI_ADAPTIVE_JACOBIAN_H
#define __NEW_MULTI_ADAPTIVE_JACOBIAN_H

#include <dolfin/TimeSlabJacobian.h>

namespace dolfin
{
  
  class MultiAdaptiveNewtonSolver;
  class MultiAdaptiveTimeSlab;
    
  /// This class represents the Jacobian matrix of the system of
  /// equations defined on a multi-adaptive time slab.

  class NewMultiAdaptiveJacobian : public TimeSlabJacobian
  {
  public:

    /// Constructor
    NewMultiAdaptiveJacobian(MultiAdaptiveNewtonSolver& newton,
			     MultiAdaptiveTimeSlab& timeslab);

    /// Destructor
    ~NewMultiAdaptiveJacobian();

    /// Compute product y = Ax
    void mult(const Vector& x, Vector& y) const;

    /// Recompute Jacobian if necessary
    void update();

    /// Friends
    friend class MultiAdaptivePreconditioner;

  private:

    // The Newton solver
    MultiAdaptiveNewtonSolver& newton;

    // The time slab
    MultiAdaptiveTimeSlab& ts;

    // Temporary storage
    real* tmp;

    // Current size of system
    uint nj;

    // Size of increment
    real h;
    
  };

}

#endif
