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
    enum State { undamped, scalar_damping, diagonal_damping };

    // Check if the time slab has converged
    bool converged(TimeSlab& timeslab);

    // Update time slab
    void update(TimeSlab& timeslab);

    // Compute stabilization
    void stabilize(TimeSlab& timeslab);

    // Compute stabilization for undamped state
    void stabilizeUndamped(TimeSlab& timeslab, real rho);

    // Compute stabilization for scalar damping
    void stabilizeScalar(TimeSlab& timeslab, real rho);

    // Compute stabilization for diagonal damping
    void stabilizeDiagonal(TimeSlab& timeslab, real rho);

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
    real d0, d1;

    // Discrete residuals
    real r0, r1;

  };

}

#endif
