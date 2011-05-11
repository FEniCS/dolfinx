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
// First added:  2005-01-27
// Last changed: 2006-08-08

#ifndef __MULTI_ADAPTIVE_FIXED_POINT_SOLVER_H
#define __MULTI_ADAPTIVE_FIXED_POINT_SOLVER_H

#include <dolfin/common/types.h>
#include <dolfin/common/real.h>
#include "TimeSlabSolver.h"

namespace dolfin
{
  class MultiAdaptiveTimeSlab;

  /// This class implements fixed-point iteration on multi-adaptive
  /// time slabs. In each iteration, the solution is updated according
  /// to the fixed-point iteration x = g(x). The iteration is
  /// performed forward in time Gauss-Seidel style, i.e., the degrees
  /// of freedom on an element are updated according to x = g(x) and
  /// the new values are used when updating the remaining elements.

  class MultiAdaptiveFixedPointSolver : public TimeSlabSolver
  {
  public:

    /// Constructor
    MultiAdaptiveFixedPointSolver(MultiAdaptiveTimeSlab& timeslab);

    /// Destructor
    ~MultiAdaptiveFixedPointSolver();

  protected:

    /// Retry solution of system by switching strategy
    bool retry();

    // Start iterations
    void start();

    // End iterations
    void end();

    // Make an iteration
    real iteration(const real& tol, uint iter, const real& d0, const real& d1);

    /// Size of system
    uint size() const;

  private:

    // The time slab
    MultiAdaptiveTimeSlab& ts;

    // Values of right-hand side at quadrature points
    real* f;

    // Total number of elements
    uint num_elements;

    // Estimated number of elements for mono-adaptive system
    real num_elements_mono;

    // Maximum number of local iterations
    uint maxiter_local;

    // True if we should use diagonal Newton damping
    bool diagonal_newton_damping;

    // Diagonal of Jacobian df/du
    real* dfdu;

  };

}

#endif
