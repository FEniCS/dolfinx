// Copyright (C) 2005 Anders Logg
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
// First added:  2005-01-05
// Last changed: 2005-11-11

#ifndef __TIME_SLAB_SOLVER_H
#define __TIME_SLAB_SOLVER_H

#include "ODE.h"

namespace dolfin
{
  class Method;
  class TimeSlab;

  /// This is the base class for solvers of the system of equations
  /// defined on time slabs.

  class TimeSlabSolver
  {
  public:

    /// Constructor
    TimeSlabSolver(TimeSlab& timeslab);

    /// Destructor
    virtual ~TimeSlabSolver();

    /// Solve system
    bool solve();

  protected:

    /// Solve system
    bool solve(uint attempt);

    /// Retry solution of system, perhaps with a new strategy (optional)
    virtual bool retry();

    /// Start iterations (optional)
    virtual void start();

    /// End iterations (optional)
    virtual void end();

    /// Make an iteration, return increment
    virtual real iteration(const real& tol, uint iter, const real& d0, const real& d1) = 0;

    /// Size of system
    virtual uint size() const = 0;

    // The ODE
    ODE& ode;

    // The method
    const Method& method;

    // Tolerance for iterations (max-norm)
    real tol;

    // Maximum number of iterations
    uint maxiter;

    // True if we should monitor the convergence
    bool monitor;

    // Number of time slabs systems solved
    uint num_timeslabs;

    // Number of global iterations made
    uint num_global_iterations;

    // Number of local iterations made (GMRES)
    uint num_local_iterations;

    // Current maxnorm of solution
    real xnorm;

  private:

    /// Choose tolerance
    void choose_tolerance();

  };

}

#endif
