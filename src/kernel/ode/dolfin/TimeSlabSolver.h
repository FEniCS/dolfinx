// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __TIME_SLAB_SOLVER_H
#define __TIME_SLAB_SOLVER_H

namespace dolfin
{

  class NewTimeSlab;
  class NewMethod;
  
  class TimeSlabSolver
  {
  public:
    
    /// Constructor
    TimeSlabSolver(NewTimeSlab& timeslab, const NewMethod& method);

    /// Destructor
    virtual ~TimeSlabSolver();

    /// Solve system
    void solve();

  protected:

    /// Iteration
    virtual real iteration() = 0;

    NewTimeSlab& ts;
    const NewMethod& method;
    
    real tol;
    uint maxiter;

  };

}

#endif
