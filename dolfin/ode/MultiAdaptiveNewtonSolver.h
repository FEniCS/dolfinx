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

#ifndef __MULTI_ADAPTIVE_NEWTON_SOLVER_H
#define __MULTI_ADAPTIVE_NEWTON_SOLVER_H

#include <boost/scoped_ptr.hpp>
#include <dolfin/common/types.h>
#include <dolfin/common/real.h>
//#include <dolfin/la/uBLASKrylovSolver.h>
#include <dolfin/la/uBLASVector.h>
#include "MultiAdaptivePreconditioner.h"
#include "TimeSlabJacobian.h"
#include "TimeSlabSolver.h"

namespace dolfin
{

  class ODE;
  class MultiAdaptiveTimeSlab;
  class Method;
  class uBLASKrylovSolver;

  /// This class implements Newton's method on multi-adaptive time
  /// slabs. In each iteration, the system F(x) is evaluated at the
  /// current solution and then the linear system A dx = b is solved
  /// for the increment dx with A = F' the Jacobian of F and b = -F(x)

  class MultiAdaptiveNewtonSolver : public TimeSlabSolver
  {
  public:

    /// Constructor
    MultiAdaptiveNewtonSolver(MultiAdaptiveTimeSlab& timeslab);

    /// Destructor
    ~MultiAdaptiveNewtonSolver();

    /// Friends
    friend class MultiAdaptiveJacobian;
    friend class UpdatedMultiAdaptiveJacobian;

  protected:

    /// Start iterations
    void start();

    // End iterations
    void end();

    // Make an iteration
    real iteration(const real& tol, uint iter, const real& d0, const real& d1);

    /// Size of system
    uint size() const;

  private:

    // Evaluate -F(x) at current x
    void Feval(uBLASVector& F);

    // Numerical evaluation of the Jacobian used for testing
    void debug();

    MultiAdaptiveTimeSlab& ts;       // The time slab;
    TimeSlabJacobian* A;             // Jacobian of time slab system
    MultiAdaptivePreconditioner mpc; // Preconditioner
    boost::scoped_ptr<uBLASKrylovSolver> solver;        // Linear solver
    real* f;                         // Values of right-hand side at quadrature points
    real* u;                         // Degrees of freedom on local element
    uBLASVector dx;                  // Increment for Newton's method
    uBLASVector b;                   // Right-hand side -F(x)
    uint num_elements;               // Total number of elements
    real num_elements_mono;          // Estimated number of elements for mono-adaptive system
    bool updated_jacobian;           // Update Jacobian in each iteration

  };

}

#endif
