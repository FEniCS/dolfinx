// Copyright (C) 2004 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2004.

#ifndef __ADAPTIVE_ITERATION_H
#define __ADAPTIVE_ITERATION_H

#include <dolfin/NewArray.h>
#include <dolfin/Iteration.h>

namespace dolfin
{

  /// State-specific behavior of fixed point iteration for general stiff problems.

  class AdaptiveIteration : public Iteration
  {
  public:

    AdaptiveIteration(Solution& u, RHS& f, FixedPointIteration& fixpoint,
		      real maxdiv, real maxconv, real tol);
    
    ~AdaptiveIteration();
    
    State state() const;

    void update(TimeSlab& timeslab, const Damping& d);
    void update(Element& element, const Damping& d);
    void update(NewArray<Element*>& elements, const Damping& d);
    
    void stabilize(TimeSlab& timeslab, const Residuals& r, Damping& d);
    void stabilize(NewArray<Element*>& elements, const Residuals& r, Damping& d);
    void stabilize(Element& element, const Residuals& r, Damping& d);
    
    bool converged(TimeSlab& timeslab, Residuals& r, unsigned int n);
    bool converged(NewArray<Element*>& elements, Residuals& r, unsigned int n);
    bool converged(Element& element, Residuals& r, unsigned int n);

    bool diverged(TimeSlab& timeslab, Residuals& r, unsigned int n, Iteration::State& newstate);
    bool diverged(NewArray<Element*>& elements, Residuals& r, unsigned int n, Iteration::State& newstate);
    bool diverged(Element& element, Residuals& r, unsigned int n, Iteration::State& newstate);

    void report() const;

  protected:
    
    real* newvalues;
    unsigned int offset;
    
  };

}

#endif
