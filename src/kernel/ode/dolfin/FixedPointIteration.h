// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FIXED_POINT_ITERATION_H
#define __FIXED_POINT_ITERATION_H

#include <dolfin/Message.h>
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

    /// Reset a given element
    void reset(Element& element);

  private:

    // Current state
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

    // Messages
    Message message_diagonal_damping;
    Message message_accelerating;
    Message message_scalar_damping;
    Message message_resetting_element;
    Message message_resetting_timeslab;
    Message message_nonconverging;

    // Maximum number of iterations
    unsigned int maxiter;

    // Maximum number of local iterations
    unsigned int local_maxiter;

    // Maximum allowed divergence
    real maxdiv;

    // Tolerance for discrete residual
    real tol;

    //--- Temporary data (cleared between iterations)

    // Current state
    State state;

    // Iteration number
    unsigned int n;    

    // Remaining number of iterations with small alpha
    unsigned int m;

    // Current damping
    real alpha;

    // Increments
    real d1, d2;

    // Discrete residuals
    real r0, r1, r2;

  };

}

#endif
