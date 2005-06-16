// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __MULTI_ADAPTIVE_NEWTON_SOLVER_H
#define __MULTI_ADAPTIVE_NEWTON_SOLVER_H

#include <dolfin/constants.h>
#include <dolfin/GMRES.h>
#include <dolfin/Vector.h>
#include <dolfin/MultiAdaptiveJacobian.h>
#include <dolfin/MultiAdaptivePreconditioner.h>
#include <dolfin/TimeSlabSolver.h>

namespace dolfin
{

  class ODE;
  class MultiAdaptiveTimeSlab;
  class Method;

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

    /// Solve system
    void solve();

  protected:

    /// Start iterations
    void start();

    // End iterations
    void end();
    
    // Make an iteration
    real iteration();

    /// Size of system
    uint size() const;

  private:

    // Evaluate b = -F(x) at current x
    void beval();

    // Numerical evaluation of the Jacobian used for testing
    void debug();

    MultiAdaptiveTimeSlab& ts;       // The time slab;
    MultiAdaptiveJacobian A;         // Jacobian of time slab system
    real* f;                         // Values of right-hand side at quadrature points
    Vector dx;                       // Increment for Newton's method
    Vector b;                        // Right-hand side -F(x)
    GMRES solver;                    // GMRES solver
    MultiAdaptivePreconditioner mpc; // Preconditioner
    uint num_elements;               // Total number of elements
    real num_elements_mono;          // Estimated number of elements for mono-adaptive system
    
  };

}

#endif
