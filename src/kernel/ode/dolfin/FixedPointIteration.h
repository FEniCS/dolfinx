// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FIXED_POINT_ITERATION_H
#define __FIXED_POINT_ITERATION_H

#include <dolfin/constants.h>
#include <dolfin/NewArray.h>
#include <dolfin/Event.h>
#include <dolfin/Iteration.h>

namespace dolfin
{
  class Solution;
  class RHS;
  class TimeSlab;
  class Element;
  class Iteration;

  /// Damped fixed point iteration on a time slab.
  ///
  /// The fixed point iteration is implemented as a state machine,
  /// with the class FixedPointIteration working as interface to the
  /// outside world ("context") and the class Iteration working as a
  /// base class ("state") for the different subclasses implementing
  /// state-specific behavior.

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
    real residual(NewArray<Element*>& elements);

    // Compute discrete residual for element
    real residual(Element& element);
    
    /// Update initial data for element list
    void init(NewArray<Element*>& elements);

    // Update initial data for element
    void init(Element& element);

    /// Reset element list
    void reset(NewArray<Element*>& elements);

    // Reset element
    void reset(Element& element);

    /// Display a status report
    void report() const;

  private:

    // Update time slab
    void update(TimeSlab& timeslab);
    
    // Update element list
    void update(NewArray<Element*>& elements);

    // Update element
    void update(Element& element);

    // Stabilize time slab
    void stabilize(TimeSlab& timeslab, const Iteration::Residuals& r, Iteration::Damping& d);

    // Stabilize element list
    void stabilize(NewArray<Element*>& elements, const Iteration::Residuals& r, Iteration::Damping& d);

    // Stabilize element
    void stabilize(Element& element, const Iteration::Residuals& r, Iteration::Damping& d);

    // Check convergence for time slab
    bool converged(TimeSlab& timeslab, Iteration::Residuals& r, unsigned int n);

    // Check convergence for element list
    bool converged(NewArray<Element*>& elements, Iteration::Residuals& r, unsigned int n);

    // Check convergence for element
    bool converged(Element& element, Iteration::Residuals& r, unsigned int n);

    // Change state
    void changeState(Iteration::State newstate);

    //--- Data for fixed point iteration

    // Solution
    Solution& u;

    // Right-hand side f
    RHS& f;

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
    Iteration* state;

  };

}

#endif
