// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-01-27
// Last changed: 2005

#ifndef __MULTI_ADAPTIVE_FIXED_POINT_SOLVER_H
#define __MULTI_ADAPTIVE_FIXED_POINT_SOLVER_H

#include <dolfin/constants.h>
#include <dolfin/TimeSlabSolver.h>

namespace dolfin
{

  class MultiAdaptiveTimeSlab;
  
  /// This class implements fixed point iteration on multi-adaptive
  /// time slabs. In each iteration, the solution is updated according
  /// to the fixed point iteration x = g(x). The iteration is
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

    /// Solve system
    void solve();

  protected:

    // Start iterations
    void start();

    // End iterations
    void end();

    // Make an iteration
    real iteration(uint iter, real tol);

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
    
    // Total number of scaled local iterations
    real num_iterations_local;

  };

}

#endif
