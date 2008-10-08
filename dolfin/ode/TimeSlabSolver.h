// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-01-05
// Last changed: 2005-11-11

#ifndef __TIME_SLAB_SOLVER_H
#define __TIME_SLAB_SOLVER_H

#include "ODE.h"

namespace dolfin
{
  class Method;
  class TimeSlab;
  
  /// This is the base class for solvers of the system of equations
  /// defined on time slabs.

  class TimeSlabSolver
  {
  public:
    
    /// Constructor
    TimeSlabSolver(TimeSlab& timeslab);

    /// Destructor
    virtual ~TimeSlabSolver();

    /// Solve system
    bool solve();

  protected:

    /// Solve system
    bool solve(uint attempt);

    /// Retry solution of system, perhaps with a new strategy (optional)
    virtual bool retry();

    /// Start iterations (optional)
    virtual void start();

    /// End iterations (optional)
    virtual void end();

    /// Make an iteration, return increment
    virtual double iteration(double tol, uint itera, double d0, double d1) = 0;

    /// Size of system
    virtual uint size() const = 0;

    // The ODE
    ODE& ode;

    // The method
    const Method& method;
    
    // Tolerance for iterations (max-norm)
    double tol;

    // Maximum number of iterations
    uint maxiter;

    // True if we should monitor the convergence
    bool monitor;

    // Number of time slabs systems solved
    uint num_timeslabs;

    // Number of global iterations made
    uint num_global_iterations;

    // Number of local iterations made (GMRES)
    uint num_local_iterations;

    // Current maxnorm of solution
    double xnorm;

  private:

    /// Choose tolerance
    void chooseTolerance();

  };

}

#endif
