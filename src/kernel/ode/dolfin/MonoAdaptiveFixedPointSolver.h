// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __MONO_ADAPTIVE_FIXED_POINT_SOLVER_H
#define __MONO_ADAPTIVE_FIXED_POINT_SOLVER_H

#include <dolfin/constants.h>
#include <dolfin/TimeSlabSolver.h>

namespace dolfin
{

  class MonoAdaptiveTimeSlab;
  
  /// This class implements fixed point iteration on mono-adaptive
  /// time slabs. In each iteration, the solution is updated according
  /// to the fixed point iteration x = g(x).

  class MonoAdaptiveFixedPointSolver : public TimeSlabSolver
  {
  public:

    /// Constructor
    MonoAdaptiveFixedPointSolver(MonoAdaptiveTimeSlab& timeslab);

    /// Destructor
    ~MonoAdaptiveFixedPointSolver();

    /// Solve system
    void solve();

  protected:

    // Make an iteration
    real iteration();

  private:

    // The time slab
    MonoAdaptiveTimeSlab& ts;

  };

}

#endif
