// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEWTON_ITERATION_H
#define __NEWTON_ITERATION_H

#include <dolfin/Iteration.h>

namespace dolfin
{

  /// Fixed-point iteration using Newton's method

  class NewtonIteration : public Iteration
  {
  public:

    NewtonIteration(Solution& u, RHS& f, FixedPointIteration& fixpoint,
		    unsigned int maxiter, real maxdiv, real maxconv,
		    real tol, unsigned int depth);

    ~NewtonIteration();

    State state() const;

    void start(ElementGroupList& list);
    void start(ElementGroup& group);
    void start(Element& element);
    
    void update(ElementGroupList& list, Increments& d);
    void update(ElementGroup& group, Increments& d);
    void update(Element& element, Increments& d);
    
    void stabilize(ElementGroupList& list, const Increments& d, unsigned int n);
    void stabilize(ElementGroup& group, const Increments& d, unsigned int n);
    void stabilize(Element& element, const Increments& d, unsigned int n);
    
    bool converged(ElementGroupList& list, const Increments& d, unsigned int n);
    bool converged(ElementGroup& group, const Increments& d, unsigned int n);
    bool converged(Element& element, const Increments& d, unsigned int n);

    bool diverged(ElementGroupList& list, const Increments& d,
		  unsigned int n, State& newstate);
    bool diverged(ElementGroup& group, const Increments& d, unsigned int n,
		  State& newstate);
    bool diverged(Element& element, const Increments& d, unsigned int n,
		  State& newstate);

    void report() const;

  };

}

#endif
