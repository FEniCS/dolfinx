// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ITERATION_H
#define __ITERATION_H

#include <dolfin/dolfin_log.h>
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

    // Element residuals
    struct Residuals
    {
      Residuals() : r0(0), r1(0), r2(0) {}
      void operator=(real r) { r1 = r2; r2 = r; }
      void reset() { r0 = 0.0; r1 = 0.0; r2 = 0.0; }
      friend LogStream& operator<<(LogStream& stream, const Residuals& r)
      { stream << "residual: " << r.r1 << " --> " << r.r2; return stream; }
      real r0, r1, r2;
    };

    // Increments
    struct Increments
    {
      Increments() : d1(0), d2(0), dtot(0) {}
      void operator=(real d) { d1 = d2; d2 = d; dtot = std::max(dtot, d); }
      void reset() { d1 = 0.0; d2 = 0.0; dtot = 0.0; }
      friend LogStream& operator<<(LogStream& stream, const Increments& d)
      { stream << "increment: " << d.d1 << " --> " << d.d2; return stream; }
      real d1, d2, dtot;
    };

    /// Constructor
    Iteration(Solution& u, RHS& f, FixedPointIteration& fixpoint,
	      unsigned int maxiter, real tol, real maxdiv, real maxconv,
	      unsigned int depth);
    
    /// Destructor
    virtual ~Iteration();

    /// Return current state (type of iteration)
    virtual State state() const = 0;

    // Start iteration on group list
    virtual void start(ElementGroupList& list) = 0;

    // Start iteration on element group
    virtual void start(ElementGroup& group) = 0;
    
    // Start iteration on element
    virtual void start(Element& element) = 0;

    /// Update group list
    virtual void update(ElementGroupList& list, Increments& d) = 0;

    /// Update element group
    virtual void update(ElementGroup& group, Increments& d) = 0;

    /// Update element
    virtual void update(Element& element, Increments& d) = 0;

    /// Stabilize group list iteration
    virtual void stabilize(ElementGroupList& list, 
			   const Residuals& r, unsigned int n) = 0;
    
    /// Stabilize element group iteration
    virtual void stabilize(ElementGroup& group,
			   const Residuals& r, unsigned int n) = 0;
    
    /// Stabilize element iteration
    virtual void stabilize(Element& element,
			   const Residuals& r, unsigned int n) = 0;
    
    /// Check convergence for group list
    virtual bool converged(ElementGroupList& list, Residuals& r, const Increments& d, unsigned int n) = 0;

    /// Check convergence for element group
    virtual bool converged(ElementGroup& group, Residuals& r,  const Increments& d, unsigned int n) = 0;

    /// Check convergence for element
    virtual bool converged(Element& element, Residuals& r, const Increments& d, unsigned int n) = 0;

    /// Check divergence for group list
    virtual bool diverged(ElementGroupList& list, const Residuals& r, const Increments& d, unsigned int n, State& newstate) = 0;
    
    /// Check divergence for element group
    virtual bool diverged(ElementGroup& group, const Residuals& r, const Increments& d, unsigned int n, State& newstate) = 0;

    /// Check divergence for element
    virtual bool diverged(Element& element, const Residuals& r, const Increments& d, unsigned int n, State& newstate) = 0;

    /// Write a status report
    virtual void report() const = 0;

    /// Return stabilization parameters
    void stabilization(real& alpha, unsigned int& m) const;

    /// Return current depth
    unsigned int depth() const;

    /// Increase depth
    void down();

    /// Decrease depth
    void up();

    /// Update initial data for element group
    void init(ElementGroup& group);

    // Update initial data for element
    void init(Element& element);

    /// Reset group list
    void reset(ElementGroupList& list);

    /// Reset element group
    void reset(ElementGroup& group);

    // Reset element
    void reset(Element& element);

    // Compute L2 norm of element residual for group list
    real residual(ElementGroupList& list);

    // Compute L2 norm of element residual for element group
    real residual(ElementGroup& group);

    // Compute absolute value of element residual for element
    real residual(Element& element);

  protected:

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
    bool stabilize(const Residuals& r, unsigned int n);
    
    // Compute divergence
    real computeDivergence(ElementGroupList& list, const Residuals& r);

    // Compute divergence
    real computeDivergence(ElementGroup& group, const Residuals& r);

    // Compute alpha
    real computeAlpha(real rho) const;

    // Compute number of damping steps
    unsigned int computeSteps(real rho) const;
    
    // Initialize data for propagation of initial values
    void initInitialData(real t0);

    // Initialize data to new size
    void initData(Values& values, unsigned int size);

    // Compute size of data
    unsigned int dataSize(ElementGroupList& list);

    // Compute size of data
    unsigned int dataSize(ElementGroup& group);

    // Copy data from group list
    void copyData(ElementGroupList& list, Values& values);

    // Copy data to group list
    void copyData(Values& values, ElementGroupList& list);

    // Copy data from element group
    void copyData(ElementGroup& group, Values& values);
    
    // Copy data to element group
    void copyData(Values& values, ElementGroup& group);

    //--- Iteration data ---

    Solution& u;
    RHS& f;
    FixedPointIteration& fixpoint;

    unsigned int maxiter;

    real maxdiv;
    real maxconv;
    real tol;

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

    // Depth of iteration (0,1,2,3)
    unsigned int _depth;

    // Temporary data used to restore element values after iteration
    Values x0;

    // Temporary data used for Gauss-Jacobi iteration
    Values x1;

    // Temporary data used for propagation of initial values
    Values u0;

  };

}

#endif
