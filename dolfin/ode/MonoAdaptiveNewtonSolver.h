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

#ifndef __MONO_ADAPTIVE_NEWTON_SOLVER_H
#define __MONO_ADAPTIVE_NEWTON_SOLVER_H

#include <dolfin/common/types.h>
#include <dolfin/la/uBLASVector.h>
#include "MonoAdaptiveJacobian.h"
#include "TimeSlabSolver.h"

namespace dolfin
{

  class uBLASKrylovSolver;
  class UmfpackLUSolver;
  class KrylovSolver;
  class LUSolver;
  class ODE;
  class MonoAdaptiveTimeSlab;
  class NewMethod;

  /// This class implements Newton's method on mono-adaptive time
  /// slabs. In each iteration, the system F(x) is evaluated at the
  /// current solution and then the linear system A dx = b is solved
  /// for the increment dx with A = F' the Jacobian of F and b = -F(x)

  class MonoAdaptiveNewtonSolver : public TimeSlabSolver
  {
  public:

    /// Constructor
    MonoAdaptiveNewtonSolver(MonoAdaptiveTimeSlab& timeslab, bool implicit = false);

    /// Destructor
    ~MonoAdaptiveNewtonSolver();

  protected:

    /// Start iterations (optional)
    void start();

    // Make an iteration
    real iteration(const real& tol, uint iter, const real& d0, const real& d1);

    /// Size of system
    uint size() const;

  private:

    // Evaluate -F(x) at current x
    void Feval(real* F);

    // Evaluate -F(x) for explicit system: u' = f
    void FevalExplicit(real* F);

    // Evaluate -F(x) for implicit system: Mu' = f
    void FevalImplicit(real* F);

    // Choose  linear solver
    void chooseLinearSolver();

    // Numerical evaluation of the Jacobian used for testing
    void debug();

    bool implicit;  // True if ODE is implicit
    bool piecewise; // True if M is piecewise constant

    MonoAdaptiveTimeSlab& ts;    // The time slab;
    MonoAdaptiveJacobian A;      // Jacobian of time slab system
    uBLASVector dx;              // Increment for Newton's method
    uBLASVector b;               // Right-hand side b = -F(x)
    real* btmp;                // Copy of right-hand side b = -F(x)
    Array<real> Mu0;                 // Precomputed product M*u0 for implicit system
    uBLASKrylovSolver* krylov;   // Iterative linear solver
    UmfpackLUSolver* lu;         // Direct linear solver
    KrylovSolver* krylov_g;      // Iterative linear solver (general)
    LUSolver* lu_g;              // Direct linear solver (general)

  };

}

#endif
