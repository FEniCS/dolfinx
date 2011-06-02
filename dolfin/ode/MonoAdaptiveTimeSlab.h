// Copyright (C) 2005-2008 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2005-01-28
// Last changed: 2009-09-08

#ifndef __MONO_ADAPTIVE_TIME_SLAB_H
#define __MONO_ADAPTIVE_TIME_SLAB_H

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/types.h>
#include "MonoAdaptivity.h"
#include "TimeSlab.h"

namespace dolfin
{

  class ODE;
  class Method;
  class TimeSlabSolver;
  class ODESolution;

  /// This class represents a mono-adaptive time slab of length k = b - a
  /// holding the degrees of freedom for the solution of an ODE between
  /// two time levels a and b.

  class MonoAdaptiveTimeSlab : public TimeSlab
  {
  public:

    /// Constructor
    MonoAdaptiveTimeSlab(ODE& ode);

    /// Destructor
    ~MonoAdaptiveTimeSlab();

    /// Build time slab, return end time
    real build(real a, real b);

    /// Solve time slab system
    bool solve();

    /// Check if current solution can be accepted
    bool check(bool first);

    /// Shift time slab (prepare for next time slab)
    bool shift(bool end);

    /// Prepare sample at time t
    void sample(real t);

    /// Sample solution value of given component at given time
    real usample(uint i, real t);

    /// Sample time step size for given component at given time
    real ksample(uint i, real t);

    /// Sample residual for given component at given time
    real rsample(uint i, real t);

    /// Save to ODESolution object
    void save_solution(ODESolution& u);

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Friends
    friend class MonoAdaptiveFixedPointSolver;
    friend class MonoAdaptiveNewtonSolver;
    friend class MonoAdaptiveJacobian;

  private:

    // Evaluate right-hand side at given quadrature point
    void feval(uint m);

    // Choose solver
    TimeSlabSolver* choose_solver();

    // Temporary data array used to store multiplications
    real* tmp();

    TimeSlabSolver* solver;    // The solver
    MonoAdaptivity adaptivity; // Adaptive time step regulation
    uint nj;                   // Number of dofs
    real* dofs;              // Local dofs for an element used for interpolation
    real* fq;                // Values of right-hand side at all quadrature points
    real rmax;               // Previously computed maximum norm of residual

    real* x; // Degrees of freedom for the solution on the time slab
    Array<real> u; // The solution at a given stage
    Array<real> f; // The right-hand side at a given stage

  };

}

#endif
