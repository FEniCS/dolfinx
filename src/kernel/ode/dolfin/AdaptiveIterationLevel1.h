// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ADAPTIVE_ITERATION_LEVEL_1_H
#define __ADAPTIVE_ITERATION_LEVEL_1_H

#include <dolfin/Iteration.h>

namespace dolfin
{

  /// State-specific behavior of fixed point iteration for stiff (level 1) problems.
  /// Adaptive damping used on the element level.

  class AdaptiveIterationLevel1 : public Iteration
  {
  public:

    AdaptiveIterationLevel1(Solution& u, RHS& f, FixedPointIteration& fixpoint,
			    unsigned int maxiter, real maxdiv, real maxconv, real tol);
    
    ~AdaptiveIterationLevel1();
    
    State state() const;
    
    void start(ElementGroupList& groups);
    void start(ElementGroup& group);
    void start(Element& element);
    
    void update(ElementGroupList& groups);
    void update(ElementGroup& group);
    void update(Element& element);    

    void stabilize(ElementGroupList& groups, const Residuals& r, unsigned int n);
    void stabilize(ElementGroup& group, const Residuals& r, unsigned int n);
    void stabilize(Element& element, const Residuals& r, unsigned int n);
    
    bool converged(ElementGroupList& groups, Residuals& r, unsigned int n);
    bool converged(ElementGroup& group, Residuals& r, unsigned int n);
    bool converged(Element& element, Residuals& r, unsigned int n);

    bool diverged(ElementGroupList& groups, Residuals& r, unsigned int n, Iteration::State& newstate);
    bool diverged(ElementGroup& group, Residuals& r, unsigned int n, Iteration::State& newstate);
    bool diverged(Element& element, Residuals& r, unsigned int n, Iteration::State& newstate);

    void report() const;

  };

}

#endif
