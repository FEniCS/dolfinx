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
			    unsigned int maxiter, real maxdiv, real maxconv, real tol,
			    unsigned int depth);
    
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

    // Simple Gauss-Jacobi update of element group
    void updateGaussJacobi(ElementGroupList& list);

    // Update group with propagation of initial values (Gauss-Seidel in time)
    void updateGaussSeidel(ElementGroupList& list);

  };

}

#endif
