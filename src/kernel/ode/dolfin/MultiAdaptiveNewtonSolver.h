// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-01-27
// Last changed: 2006-05-07

#ifndef __MULTI_ADAPTIVE_NEWTON_SOLVER_H
#define __MULTI_ADAPTIVE_NEWTON_SOLVER_H

#ifdef HAVE_PETSC_H

#include <dolfin/constants.h>
#include <dolfin/GMRES.h>
#include <dolfin/Vector.h>
#include <dolfin/MultiAdaptivePreconditioner.h>
#include <dolfin/TimeSlabJacobian.h>
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

    /// Friends
    friend class MultiAdaptiveJacobian;
    friend class UpdatedMultiAdaptiveJacobian;

  protected:

    /// Start iterations
    void start();

    // End iterations
    void end();
    
    // Make an iteration
    real iteration(uint iter, real tol);

    /// Size of system
    uint size() const;

  private:

    // Evaluate -F(x) at current x
    void Feval(real F[]);

    // Numerical evaluation of the Jacobian used for testing
    void debug();

    MultiAdaptiveTimeSlab& ts;       // The time slab;
    TimeSlabJacobian* A;             // Jacobian of time slab system
    MultiAdaptivePreconditioner mpc; // Preconditioner
    GMRES solver;                    // GMRES solver
    real* f;                         // Values of right-hand side at quadrature points
    Vector dx;                       // Increment for Newton's method
    Vector b;                        // Right-hand side -F(x)
    uint num_elements;               // Total number of elements
    real num_elements_mono;          // Estimated number of elements for mono-adaptive system
    bool updated_jacobian;           // Update Jacobian in each iteration

  };

}

#endif

#endif
