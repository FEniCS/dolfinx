// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ITERATION_H
#define __ITERATION_H

#include <dolfin/constants.h>

namespace dolfin
{
  class Solution;
  class RHS;
  class TimeSlab;
  class Element;
  class ElementGroup;
  class ElementGroupList;
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

    // Start iteration on element group list
    virtual void start(ElementGroupList& list) = 0;

    // Start iteration on element group
    virtual void start(ElementGroup& group) = 0;
    
    // Start iteration on element
    virtual void start(Element& element) = 0;

    /// Update element group list
    virtual void update(ElementGroupList& list) = 0;

    /// Update element
    virtual void update(Element& element) = 0;

    /// Update element group
    virtual void update(ElementGroup& group) = 0;

    /// Stabilize element group list iteration
    virtual void stabilize(ElementGroupList& list, 
			   const Residuals& r, unsigned int n) = 0;
    
    /// Stabilize element group iteration
    virtual void stabilize(ElementGroup& group,
			   const Residuals& r, unsigned int n) = 0;
    
    /// Stabilize element iteration
    virtual void stabilize(Element& element,
			   const Residuals& r, unsigned int n) = 0;
    
    /// Check convergence for element group list
    virtual bool converged(ElementGroupList& list, Residuals& r, unsigned int n) = 0;

    /// Check convergence for element group
    virtual bool converged(ElementGroup& group, Residuals& r, unsigned int n) = 0;

    /// Check convergence for element
    virtual bool converged(Element& element, Residuals& r, unsigned int n) = 0;

    /// Check divergence for element group list
    virtual bool diverged(ElementGroupList& list, Residuals& r, unsigned int n, Iteration::State& newstate) = 0;

    /// Check divergence for element group
    virtual bool diverged(ElementGroup& group, Residuals& r, unsigned int n, Iteration::State& newstate) = 0;

    /// Check divergence for element
    virtual bool diverged(Element& element, Residuals& r, unsigned int n, Iteration::State& newstate) = 0;

    /// Write a status report
    virtual void report() const = 0;

    /// Update initial data for element group
    void init(ElementGroup& group);

    // Update initial data for element
    void init(Element& element);

    /// Reset element group list
    void reset(ElementGroupList& groups);

    /// Reset element group
    void reset(ElementGroup& group);

    // Reset element
    void reset(Element& element);

    // Compute L2 norm of element residual for element group list
    real residual(ElementGroupList& list);

    // Compute L2 norm of element residual for element group
    real residual(ElementGroup& group);

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
