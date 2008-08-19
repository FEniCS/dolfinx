// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-01-27
// Last changed: 2006-08-08

#ifndef __UPDATED_MULTI_ADAPTIVE_JACOBIAN_H
#define __UPDATED_MULTI_ADAPTIVE_JACOBIAN_H

#include "TimeSlabJacobian.h"

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
    uint size(uint dim) const;

    /// Compute product y = Ax
    void mult(const uBLASVector& x, uBLASVector& y) const;

    /// (Re-)initialize computation of Jacobian
    void init();

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
