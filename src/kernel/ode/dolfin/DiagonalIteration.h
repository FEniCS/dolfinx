// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __DIAGONAL_ITERATION_H
#define __DIAGONAL_ITERATION_H

#include <dolfin/NewArray.h>
#include <dolfin/Iteration.h>

namespace dolfin
{

  /// State-specific behavior of fixed point iteration for diagonally stiff problems.

  class DiagonalIteration : public Iteration
  {
  public:

    DiagonalIteration(Solution& u, RHS& f, FixedPointIteration& fixpoint,
		      real maxdiv, real maxconv, real tol);
    
    ~DiagonalIteration();

    State state() const;

    void update(TimeSlab& timeslab);
    void update(Element& element);
    void update(NewArray<Element*>& elements);
    
    State stabilize(TimeSlab& timeslab, const Residuals& r, Damping& d);
    State stabilize(NewArray<Element*>& elements, const Residuals& r, Damping& d);
    State stabilize(Element& element, const Residuals& r, Damping& d);
    
    bool converged(TimeSlab& timeslab, Residuals& r, unsigned int n);
    bool converged(NewArray<Element*>& elements, Residuals& r, unsigned int n);
    bool converged(Element& element, Residuals& r, unsigned int n);

    void report() const;

  };

}

#endif
