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
			    unsigned int depth, bool debug_iter);
    
    ~AdaptiveIterationLevel3();
    
    State state() const;

    void start(ElementGroupList& list);
    void start(ElementGroup& group);
    void start(Element& element);

    void update(ElementGroupList& list, Increments& d);
    void update(ElementGroup& group, Increments& d);
    void update(Element& element, Increments& d);
    
    void stabilize(ElementGroupList& list, const Residuals& r, const Increments& d, unsigned int n);
    void stabilize(ElementGroup& group, const Residuals& r, const Increments& d, unsigned int n);
    void stabilize(Element& element, const Residuals& r, const Increments& d, unsigned int n);
    
    bool converged(ElementGroupList& list, Residuals& r, const Increments& d, unsigned int n);
    bool converged(ElementGroup& group, Residuals& r, const Increments& d, unsigned int n);
    bool converged(Element& element, Residuals& r, const Increments& d, unsigned int n);
    
    bool diverged(ElementGroupList& list, const Residuals& r, const Increments& d,unsigned int n, State& newstate);
    bool diverged(ElementGroup& group, const Residuals& r, const Increments& d,unsigned int n, State& newstate);
    bool diverged(Element& element, const Residuals& r, const Increments& d,unsigned int n, State& newstate);
    
    void report() const;

  private:

    // Simple Gauss-Jacobi update of element group
    void updateGaussJacobi(ElementGroupList& list, Increments& d);

    // Update group with propagation of initial values (Gauss-Seidel in time)
    void updateGaussSeidel(ElementGroupList& list, Increments& d);

  };

}

#endif
