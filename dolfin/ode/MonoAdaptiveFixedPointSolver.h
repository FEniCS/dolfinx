// Copyright (C) 2005-2006 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2005-01-28
// Last changed: 2006-08-08

#ifndef __MONO_ADAPTIVE_FIXED_POINT_SOLVER_H
#define __MONO_ADAPTIVE_FIXED_POINT_SOLVER_H

#include <dolfin/common/types.h>
#include <dolfin/common/real.h>
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
    real iteration(const real& tol, uint iter, const real& d0, const real& d1);

    /// Size of system
    uint size() const;

  private:

    // The time slab
    MonoAdaptiveTimeSlab& ts;

    // Old values at right end-point used to compute the increment
    real* xold;

    // Damping (alpha = 1.0 for no damping)
    real alpha;

    // Stabilization
    bool stabilize;

    // Stabilization parameters

    // Number of stabilizing iterations
    uint mi;

    // Number of ramping iterations
    uint li;

    // Ramping coefficient
    real ramp;

    // Ramping factor
    real rampfactor;

  };

}

#endif
