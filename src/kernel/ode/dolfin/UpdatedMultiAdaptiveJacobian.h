// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-01-27
// Last changed: 2006-07-06

#ifndef __UPDATED_MULTI_ADAPTIVE_JACOBIAN_H
#define __UPDATED_MULTI_ADAPTIVE_JACOBIAN_H

#include <dolfin/TimeSlabJacobian.h>

namespace dolfin
{
  class MultiAdaptiveNewtonSolver;
  class MultiAdaptiveTimeSlab;
    
  /// This class represents the Jacobian matrix of the system of
  /// equations defined on a multi-adaptive time slab.

  class UpdatedMultiAdaptiveJacobian : public TimeSlabJacobian
  {
  public:

    /// Constructor
    UpdatedMultiAdaptiveJacobian(MultiAdaptiveNewtonSolver& newton,
			     MultiAdaptiveTimeSlab& timeslab);

    /// Destructor
    ~UpdatedMultiAdaptiveJacobian();

    /// Return number of rows (dim = 0) or columns (dim = 1)
    uint size(const uint dim) const;

    /// Compute product y = Ax
    void mult(const uBlasVector& x, uBlasVector& y) const;

    /// Recompute Jacobian if necessary
    void update();

    /// Friends
    friend class MultiAdaptivePreconditioner;

  private:

    // The Newton solver
    MultiAdaptiveNewtonSolver& newton;

    // The time slab
    MultiAdaptiveTimeSlab& ts;

    // Size of increment
    real h;
    
  };

}

#endif
