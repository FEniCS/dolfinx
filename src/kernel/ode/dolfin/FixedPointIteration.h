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
    void iterate(TimeSlab& timeslab);

    /// Update a given element
    real update(Element& element);

  private:

    // States
    enum State { undamped, scalar_small, scalar_increasing, diagonal_small, diagonal_increasing };

    // Check if the time slab has converged
    bool converged(TimeSlab& timeslab);

    // Update time slab
    void update(TimeSlab& timeslab);

    // Compute stabilization
    void stabilize(TimeSlab& timeslab);

    // Compute stabilization for undamped state
    void stabilizeUndamped(TimeSlab& timeslab, real rho);

    // Compute stabilization for scalar damping with small alpha
    void stabilizeScalarSmall(TimeSlab& timeslab, real rho);

    // Compute stabilization for scalar damping with increasing alpha
    void stabilizeScalarIncreasing(TimeSlab& timeslab, real rho);

    // Compute stabilization for diagonal damping with small alpha
    void stabilizeDiagonalSmall(TimeSlab& timeslab, real rho);

    // Compute stabilization for diagonal damping with increasing alpha
    void stabilizeDiagonalIncreasing(TimeSlab& timeslab, real rho);

    // Compute alpha
    real computeDamping(real rho);

    // Compute m
    unsigned int computeDampingSteps(real rho);

    // Clear data from previous iteration
    void clear();

    //--- Data for fixed point iteration

    // Solution
    Solution& u;

    // Right-hand side f
    RHS& f;

    // Iteration number
    unsigned int n;

    // Maximum number of iterations
    unsigned int maxiter;

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

  };

}

#endif
