// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FIXED_POINT_ITERATION_H
#define __FIXED_POINT_ITERATION_H

#include <dolfin/constants.h>
#include <dolfin/NewArray.h>
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

    /// Fixed point iteration on time slab
    bool iterate(TimeSlab& timeslab);

    /// Fixed point iteration on element list
    bool iterate(NewArray<Element*>& elements);
    
    /// Fixed point iteration on element
    bool iterate(Element& element);

    // Compute maximum discrete residual for time slab
    real residual(TimeSlab& timeslab);

    // Compute maximum discrete residual for element list
    real residual(NewArray<Element*> elements);

    // Compute discrete residual for element
    real residual(Element& element);
    
    /// Update initial data for element list
    void init(NewArray<Element*>& elements);

    /// Reset element list
    void reset(NewArray<Element*>& elements);

    /// Display a status report
    void report() const;

    /// Friends
    friend class NonStiffIteration;

  private:

    // Current state (type of problem)
    enum State { nonstiff, diagonal, parabolic, nonnormal };

    // Current damping for problems of type {parabolic}
    enum SubState { undamped, damped, increasing };

    // Discrete residuals
    struct Residuals
    {
      Residuals() : r0(0), r1(0), r2(0) {}
      real r0, r1, r2;
    };

    // Damping
    struct Damping
    {
      Damping() : alpha(0), m(0) {}
      real alpha;
      unsigned int m;
    };
    
    // Update time slab
    void update(TimeSlab& timeslab);
    
    // Update element list
    void update(NewArray<Element*>& elements);

    // Update element
    void update(Element& element);

    // Check if time slab has converged
    bool converged(TimeSlab& timeslab, Residuals& r, unsigned int n);

    // Check if element list has converged
    bool converged(NewArray<Element*>& elements, Residuals& r, unsigned int n);

    // Check if element has converged
    bool converged(Element& element, Residuals& r, unsigned int n);

    // Stabilize time slab
    void stabilize(TimeSlab& timeslab, const Residuals& r);

    // Stabilize element list
    void stabilize(NewArray<Element*>& elements, const Residuals& r);

    // Stabilize element
    void stabilize(Element& element, const Residuals& r);

    // Simple undamped update of element
    void updateUndamped(Element& element);

    // Locally damped update of element
    void updateLocalDamping(Element& element);

    // Globaly damped update of element
    void updateGlobalDamping(Element& element);

    // Compute stabilization for state {nonstiff}
    void stabilizeNonStiff(TimeSlab& timeslab, const Residuals& r);

    // Compute stabilization for state {diagonal}
    void stabilizeDiagonal(TimeSlab& timeslab, const Residuals& r);

    // Compute stabilization for state {parabolic} using damping {undamped}
    void stabilizeParabolicUndamped(TimeSlab& timeslab, const Residuals& r);

    // Compute stabilization for state {parabolic} using damping {damped}
    void stabilizeParabolicDamped(TimeSlab& timeslab, const Residuals& r);

    // Compute stabilization for state {parabolic} using damping {increasing}
    void stabilizeParabolicIncreasing(TimeSlab& timeslab, const Residuals& r);

    // Compute stabilization for state {nonnormal}
    void stabilizeNonNormal(TimeSlab& timeslab, const Residuals& r);

    // Update initial data for element
    void init(Element& element);

    // Reset element
    void reset(Element& element);

    // Compute convergence rate
    real computeConvergenceRate(const Residuals& r);

    // Compute alpha
    real computeDamping(real rho);

    // Compute m
    unsigned int computeDampingSteps(real rho);

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

    // Maximum allowed convergence
    real maxconv;

    // Tolerance for discrete residual
    real tol;

    // Current state
    State state;

    //--- Temporary data (cleared between iterations)

    // Sub state for parabolic damping
    SubState substate;

    // Remaining number of iterations with small alpha
    unsigned int m;

    // Current damping
    real alpha;

  };

}

#endif
