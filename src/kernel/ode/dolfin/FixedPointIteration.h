// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FIXED_POINT_ITERATION_H
#define __FIXED_POINT_ITERATION_H

#include <dolfin/constants.h>
#include <dolfin/Event.h>

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

    // Current state (type of problem)
    enum State { nonstiff, diagonal, parabolic, nonnormal };

    // Current damping for problems of type {parabolic}
    enum Damping { undamped, damped, increasing };

    // Update time slab
    void update(TimeSlab& timeslab);

    // Simple undamped update of element
    real updateUndamped(Element& element);

    // Locally damped update of element
    real updateLocalDamping(Element& element);

    // Globaly damped update of element
    real updateGlobalDamping(Element& element);

    // Compute stabilization
    void stabilize(TimeSlab& timeslab);

    // Compute stabilization for state {nonstiff}
    void stabilizeNonStiff(TimeSlab& timeslab);

    // Compute stabilization for state {diagonal}
    void stabilizeDiagonal(TimeSlab& timeslab);

    // Compute stabilization for state {parabolic} using damping {undamped}
    void stabilizeParabolicUndamped(TimeSlab& timeslab);

    // Compute stabilization for state {parabolic} using damping {damped}
    void stabilizeParabolicDamped(TimeSlab& timeslab);

    // Compute stabilization for state {parabolic} using damping {increasing}
    void stabilizeParabolicIncreasing(TimeSlab& timeslab);

    // Compute stabilization for state {nonnormal}
    void stabilizeNonNormal(TimeSlab& timeslab);

    // Compute convergence rate
    real computeConvergenceRate();

    // Compute alpha
    real computeDamping(real rho);

    // Compute m
    unsigned int computeDampingSteps(real rho);

    // Check if the time slab has converged
    bool converged(TimeSlab& timeslab);

    // Reset fixed point iteration
    void reset();

    //--- Data for fixed point iteration

    // Solution
    Solution& u;

    // Right-hand side f
    RHS& f;

    // Events
    Event event_diag_damping;
    Event event_accelerating;
    Event event_scalar_damping;
    Event event_reset_element;
    Event event_reset_timeslab;
    Event event_nonconverging;

    // Maximum number of iterations
    unsigned int maxiter;

    // Maximum number of local iterations
    unsigned int local_maxiter;

    // Maximum allowed divergence
    real maxdiv;

    // Tolerance for discrete residual
    real tol;

    // Current state
    State state;

    //--- Temporary data (cleared between iterations)

    // Current damping
    Damping damping;

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
