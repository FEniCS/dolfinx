// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FIXED_POINT_ITERATION_H
#define __FIXED_POINT_ITERATION_H

#include <dolfin/constants.h>

namespace dolfin
{
  class Solution;
  class RHS;
  class TimeSlab;
  class Element;

  /// Damped fixed point iteration on a time slab.

  class FixedPointIteration
  {
  public:

    /// Constructor
    FixedPointIteration(Solution& u, RHS& f);
    
    /// Destructor
    ~FixedPointIteration();

    /// Fixed point iteration on a time slab
    bool iterate(TimeSlab& timeslab);

    /// Update a given element
    real update(Element& element);

  private:

    // States
    enum State { undamped, damped, increasing };

    // Check if the time slab has converged
    bool converged(TimeSlab& timeslab);

    // Update time slab
    void update(TimeSlab& timeslab);

    // Compute stabilization
    void stabilize(TimeSlab& timeslab);

    // Compute stabilization for undamped state
    void stabilizeUndamped(TimeSlab& timeslab);

    // Compute stabilization for scalar damping with small alpha
    void stabilizeDamped(TimeSlab& timeslab);

    // Compute stabilization for scalar damping with increasing alpha
    void stabilizeIncreasing(TimeSlab& timeslab);

    // Compute convergence rate
    real computeConvergenceRate();

    // Compute alpha
    real computeDamping(real rho);

    // Compute m
    unsigned int computeDampingSteps(real rho);

    // Reset fixed point iteration
    void reset();

    //--- Data for fixed point iteration

    // Solution
    Solution& u;

    // Right-hand side f
    RHS& f;

    // Iteration number
    unsigned int n;
    
    // Maximum number of iterations
    unsigned int maxiter;

    // Maximum number of local iterations
    unsigned int local_maxiter;

    // Current state
    State state;

    // Current damping
    real alpha;

    // Remaining number of iterations with small alpha
    unsigned int m;
    
    // Increments
    real d1, d2;

    // Discrete residuals
    real r0, r1, r2;

    // Tolerance for discrete residual
    real tol;

  };

}

#endif
