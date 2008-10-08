// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-01-28
// Last changed: 2006-08-08

#ifndef __MONO_ADAPTIVE_FIXED_POINT_SOLVER_H
#define __MONO_ADAPTIVE_FIXED_POINT_SOLVER_H

#include <dolfin/common/types.h>
#include "TimeSlabSolver.h"

namespace dolfin
{
  class MonoAdaptiveTimeSlab;
  
  /// This class implements fixed-point iteration on mono-adaptive
  /// time slabs. In each iteration, the solution is updated according
  /// to the fixed-point iteration x = g(x).

  class MonoAdaptiveFixedPointSolver : public TimeSlabSolver
  {
  public:

    /// Constructor
    MonoAdaptiveFixedPointSolver(MonoAdaptiveTimeSlab& timeslab);

    /// Destructor
    ~MonoAdaptiveFixedPointSolver();

    /// Solve system
    //bool solve();

  protected:

    // Make an iteration
    double iteration(double tol, uint iter, double d0, double d1);

    /// Size of system
    uint size() const;

  private:

    // The time slab
    MonoAdaptiveTimeSlab& ts;

    // Old values at right end-point used to compute the increment
    double* xold;

    // Damping (alpha = 1.0 for no damping)
    double alpha;

    // Stabilization
    bool stabilize;

    // Stabilization parameters

    // Number of stabilizing iterations
    uint mi;

    // Number of ramping iterations
    uint li;

    // Ramping coefficient
    double ramp;

    // Ramping factor
    double rampfactor;

  };

}

#endif
