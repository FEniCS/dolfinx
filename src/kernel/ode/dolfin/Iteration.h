// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ITERATION_H
#define __ITERATION_H

#include <dolfin/constants.h>
#include <dolfin/NewArray.h>

namespace dolfin
{
  class Solution;
  class RHS;
  class TimeSlab;
  class Element;
  class FixedPointIteration;

  /// Base class for state-specific behavior of fixed point iteration.

  class Iteration
  {
  public:

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

    /// Constructor
    Iteration(Solution& u, RHS& f, FixedPointIteration& fixpoint,
	      real tol, real maxdiv, real maxconv);
    
    /// Destructor
    virtual ~Iteration();

    /// Update time slab
    virtual void update(TimeSlab& timeslab) = 0;

    /// Update element
    virtual void update(Element& element) = 0;

    /// Update element list
    virtual void update(NewArray<Element*>& elements) = 0;

    /// Stabilize time slab iteration
    virtual void stabilize(TimeSlab& timeslab, 
			   const Residuals& r) = 0;
    
    /// Stabilize element list iteration
    virtual void stabilize(NewArray<Element*>& elements,
			  const Residuals& r) = 0;
    
    /// Stabilize element iteration
    virtual void stabilize(Element& element,
			   const Residuals& r) = 0;
    
    /// Check convergence for time slab
    virtual bool converged(TimeSlab& timeslab, Residuals& r, unsigned int n) = 0;

    /// Check convergence for element list
    virtual bool converged(NewArray<Element*>& elements, Residuals& r, unsigned int n) = 0;

    /// Check convergence for element
    virtual bool converged(Element& element, Residuals& r, unsigned int n) = 0;

    /// Write a status report
    virtual void report() const = 0;

    // Compute maximum discrete residual for time slab
    real residual(TimeSlab& timeslab);

    // Compute maximum discrete residual for element list
    real residual(NewArray<Element*> elements);

    // Compute discrete residual for element
    real residual(Element& element);

    // Compute convergence rate
    real computeConvergenceRate(const Iteration::Residuals& r);

    // Compute alpha
    real computeDamping(real rho);

    // Compute m
    unsigned int computeDampingSteps(real rho);

  protected:

    Solution& u;
    RHS& f;
    FixedPointIteration& fixpoint;

    real maxdiv;
    real maxconv;
    real tol;

  };

}

#endif
