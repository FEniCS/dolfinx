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

    // Type of iteration
    enum State {nonstiff, stiff1, stiff2, stiff3, stiff};

    // Discrete residuals
    struct Residuals
    {
      Residuals() : r0(0), r1(0), r2(0) {}
      real r0, r1, r2;
    };

    /// Constructor
    Iteration(Solution& u, RHS& f, FixedPointIteration& fixpoint,
	      unsigned int maxiter, real tol, real maxdiv, real maxconv);
    
    /// Destructor
    virtual ~Iteration();

    /// Return current current state (type of iteration)
    virtual State state() const = 0;

    // Start iteration on time slab
    virtual void start(TimeSlab& timeslab) = 0;

    // Start iteration on element list
    virtual void start(NewArray<Element*>& elements) = 0;

    // Start iteration on element
    virtual void start(Element& element) = 0;

    /// Update time slab
    virtual void update(TimeSlab& timeslab) = 0;

    /// Update element
    virtual void update(Element& element) = 0;

    /// Update element list
    virtual void update(NewArray<Element*>& elements) = 0;

    /// Stabilize time slab iteration
    virtual void stabilize(TimeSlab& timeslab, 
			   const Residuals& r, unsigned int n) = 0;
    
    /// Stabilize element list iteration
    virtual void stabilize(NewArray<Element*>& elements,
			   const Residuals& r, unsigned int n) = 0;
    
    /// Stabilize element iteration
    virtual void stabilize(Element& element,
			   const Residuals& r, unsigned int n) = 0;
    
    /// Check convergence for time slab
    virtual bool converged(TimeSlab& timeslab, Residuals& r, unsigned int n) = 0;

    /// Check convergence for element list
    virtual bool converged(NewArray<Element*>& elements, Residuals& r, unsigned int n) = 0;

    /// Check convergence for element
    virtual bool converged(Element& element, Residuals& r, unsigned int n) = 0;

    /// Check divergence for time slab
    virtual bool diverged(TimeSlab& timeslab, Residuals& r, unsigned int n, Iteration::State& newstate) = 0;

    /// Check divergence for element list
    virtual bool diverged(NewArray<Element*>& elements, Residuals& r, unsigned int n, Iteration::State& newstate) = 0;

    /// Check divergence for element
    virtual bool diverged(Element& element, Residuals& r, unsigned int n, Iteration::State& newstate) = 0;

    /// Write a status report
    virtual void report() const = 0;

    /// Update initial data for element list
    void init(NewArray<Element*>& elements);

    // Update initial data for element
    void init(Element& element);

    /// Reset element list
    void reset(NewArray<Element*>& elements);

    // Reset element
    void reset(Element& element);

    // Compute L2 norm of element residual for time slab
    real residual(TimeSlab& timeslab);

    // Compute L2 norm of element residual for element list
    real residual(NewArray<Element*>& elements);

    // Compute absolute value of element residual for element
    real residual(Element& element);

  protected:

    // Type of iteration
    enum Method {gauss_jacobi, gauss_seidel};

    // An array of values used for Gauss-Jacobi iteration
    struct Values
    {
      Values();
      ~Values();

      void init(unsigned int size);

      real* values;
      unsigned int size;
      unsigned int offset;
    };

    // Stabilization for adaptive iteration
    void stabilize(const Residuals& r, real rho);

    // Compute alpha
    real computeAlpha(real rho) const;

    // Compute number of damping steps
    unsigned int computeSteps(real rho) const;

    //--- Iteration data ---

    Solution& u;
    RHS& f;
    FixedPointIteration& fixpoint;

    unsigned int maxiter;

    real maxdiv;
    real maxconv;
    real tol;

    // Current method (Gauss-Jacobi or Gauss-Seidel)
    Method method;

    // Stabilization parameter
    real alpha;

    // Angle of sector, gamma = cos(theta)
    real gamma;

    // Residual at start of stabilizing iterations
    real r0;

    // Number of stabilizing iterations
    unsigned int m;

    // Number of remaining stabilizing iterations
    unsigned int j;

  };

}

#endif
