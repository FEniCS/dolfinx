// Copyright (C) 2004 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2004.

#ifndef __ADAPTIVE_ITERATION_LEVEL_2_H
#define __ADAPTIVE_ITERATION_LEVEL_2_H

#include <dolfin/NewArray.h>
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
    void start(NewArray<Element*>& elements);
    void start(Element& element);

    void update(TimeSlab& timeslab);
    void update(NewArray<Element*>& elements);
    void update(Element& element);
    
    void stabilize(TimeSlab& timeslab, const Residuals& r, unsigned int n);
    void stabilize(NewArray<Element*>& elements, const Residuals& r, unsigned int n);
    void stabilize(Element& element, const Residuals& r, unsigned int n);
    
    bool converged(TimeSlab& timeslab, Residuals& r, unsigned int n);
    bool converged(NewArray<Element*>& elements, Residuals& r, unsigned int n);
    bool converged(Element& element, Residuals& r, unsigned int n);

    bool diverged(TimeSlab& timeslab, Residuals& r, unsigned int n, Iteration::State& newstate);
    bool diverged(NewArray<Element*>& elements, Residuals& r, unsigned int n, Iteration::State& newstate);
    bool diverged(Element& element, Residuals& r, unsigned int n, Iteration::State& newstate);

    void report() const;

  private:

    // Gauss-Jacobi iteration on element list
    void updateGaussJacobi(NewArray<Element*>& elements);

    // Gauss-Seidel iteration on element list
    void updateGaussSeidel(NewArray<Element*>& elements);
    
    // Compute divergence
    real computeDivergence(NewArray<Element*>& elements, const Residuals& r);

    // Initialize additional data
    void initData(Values& values);

    // Copy data from element list
    void copyData(const NewArray<Element*>& elements, Values& values);

    // Copy data to element list
    void copyData(const Values& values, NewArray<Element*>& elements) const;

    // Compute size of data
    unsigned int dataSize(const NewArray<Element*>& elements) const;

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
