// Copyright (C) 2004 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2004.

#ifndef __ADAPTIVE_ITERATION_LEVEL_2_H
#define __ADAPTIVE_ITERATION_LEVEL_2_H

#include <dolfin/Iteration.h>

namespace dolfin
{

  /// State-specific behavior of fixed point iteration for stiff (level 2) problems.
  /// Adaptive damping used on the element list level.

  class AdaptiveIterationLevel2 : public Iteration
  {
  public:
    
    AdaptiveIterationLevel2(Solution& u, RHS& f, FixedPointIteration& fixpoint,
			    unsigned int maxiter, real maxdiv, real maxconv, real tol);
    
    ~AdaptiveIterationLevel2();
    
    State state() const;

    void start(TimeSlab& timeslab);
    void start(ElementGroup& group);
    void start(Element& element);

    void update(TimeSlab& timeslab);
    void update(ElementGroup& group);
    void update(Element& element);
    
    void stabilize(TimeSlab& timeslab, const Residuals& r, unsigned int n);
    void stabilize(ElementGroup& group, const Residuals& r, unsigned int n);
    void stabilize(Element& element, const Residuals& r, unsigned int n);
    
    bool converged(TimeSlab& timeslab, Residuals& r, unsigned int n);
    bool converged(ElementGroup& group, Residuals& r, unsigned int n);
    bool converged(Element& element, Residuals& r, unsigned int n);

    bool diverged(TimeSlab& timeslab, Residuals& r, unsigned int n, Iteration::State& newstate);
    bool diverged(ElementGroup& group, Residuals& r, unsigned int n, Iteration::State& newstate);
    bool diverged(Element& element, Residuals& r, unsigned int n, Iteration::State& newstate);

    void report() const;

  private:

    // Gauss-Jacobi iteration on element list
    void updateGaussJacobi(ElementGroup& group);

    // Gauss-Seidel iteration on element list
    void updateGaussSeidel(ElementGroup& group);
    
    // Compute divergence
    real computeDivergence(ElementGroup& group, const Residuals& r);

    // Initialize additional data
    void initData(Values& values);

    // Copy data from element list
    void copyData(ElementGroup& group, Values& values);

    // Copy data to element list
    void copyData(Values& values, ElementGroup& group);

    // Compute size of data
    unsigned int dataSize(ElementGroup& group);

    //--- Iteration data ---
   
    // Solution values for divergence computation
    Values x0;
 
    // Solution values for Gauss-Jacobi iteration
    Values x1;
   
    // Number of values in current element list
    unsigned int datasize;

  };

}

#endif
