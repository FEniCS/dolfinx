// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __TIME_SLAB_SOLVER_H
#define __TIME_SLAB_SOLVER_H

namespace dolfin
{

  class ODE;
  class NewMethod;
  class NewTimeSlab;
  
  /// This is the base class for solvers of the system of equations
  /// defined on time slabs.

  class TimeSlabSolver
  {
  public:
    
    /// Constructor
    TimeSlabSolver(NewTimeSlab& timeslab);

    /// Destructor
    virtual ~TimeSlabSolver();

    /// Solve system
    bool solve();

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
