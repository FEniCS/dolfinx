// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ADAPTIVE_ITERATION_LEVEL_3_H
#define __ADAPTIVE_ITERATION_LEVEL_3_H

#include <dolfin/Iteration.h>

namespace dolfin
{
  
  /// State-specific behavior of fixed point iteration for stiff (level 3) problems.
  /// Adaptive damping used on the group list level (time slab level).

  class AdaptiveIterationLevel3 : public Iteration
  {
  public:
    
    AdaptiveIterationLevel3(Solution& u, RHS& f, FixedPointIteration& fixpoint,
			    unsigned int maxiter, real maxdiv, real maxconv, real tol);
    
    ~AdaptiveIterationLevel3();
    
    State state() const;

    void start(ElementGroupList& list);
    void start(ElementGroup& group);
    void start(Element& element);

    void update(ElementGroupList& list);
    void update(ElementGroup& group);
    void update(Element& element);
    
    void stabilize(ElementGroupList& list, const Residuals& r, unsigned int n);
    void stabilize(ElementGroup& group, const Residuals& r, unsigned int n);
    void stabilize(Element& element, const Residuals& r, unsigned int n);
    
    bool converged(ElementGroupList& list, Residuals& r, unsigned int n);
    bool converged(ElementGroup& group, Residuals& r, unsigned int n);
    bool converged(Element& element, Residuals& r, unsigned int n);
    
    bool diverged(ElementGroupList& list, Residuals& r, unsigned int n, Iteration::State& newstate);
    bool diverged(ElementGroup& group, Residuals& r, unsigned int n, Iteration::State& newstate);
    bool diverged(Element& element, Residuals& r, unsigned int n, Iteration::State& newstate);
    
    void report() const;
    
  private:
    
    // Gauss-Jacobi iteration on group list
    void updateGaussJacobi(ElementGroupList& list);
    
    // Compute divergence
    real computeDivergence(ElementGroupList& list, const Residuals& r);

    // Initialize additional data
    void initData(Values& values);

    // Copy data from group list
    void copyData(ElementGroupList& list, Values& values);

    // Copy data to group list
    void copyData(Values& values, ElementGroupList& list);

    // Compute size of data
    unsigned int dataSize(ElementGroupList& list);

    //--- Iteration data ---
   
    // Solution values for divergence computation
    Values x0;
 
    // Solution values for Gauss-Jacobi iteration
    Values x1;
   
    // Number of values in current element group list
    unsigned int datasize;

    // True if the element group iteration is local (and not part of group list iteration)
    bool local_iteration;

  };

}

#endif
