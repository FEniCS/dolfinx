// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __TIME_SLAB_SOLVER_H
#define __TIME_SLAB_SOLVER_H

namespace dolfin
{

  class ODE;
  class NewTimeSlab;
  class NewMethod;
  
  class TimeSlabSolver
  {
  public:
    
    /// Constructor
    TimeSlabSolver(NewTimeSlab& timeslab);

    /// Destructor
    virtual ~TimeSlabSolver();

    /// Solve system
    void solve();

  protected:

    /// Start iterations (optional)
    virtual void start();

    /// End iterations (optional)
    virtual void end();

    /// Make an iteration
    virtual real iteration() = 0;

    // The ODE
    ODE& ode;

    // The method
    const NewMethod& method;
    
    // Tolerance for iterations (max-norm)
    real tol;

    // Maximum number of iterations
    uint maxiter;

  };

}

#endif
