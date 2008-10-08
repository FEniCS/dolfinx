// Copyright (C) 2005-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-01-05
// Last changed: 2008-06-11

#include <cmath>
#include <dolfin/common/constants.h>
#include <dolfin/parameter/parameters.h>
#include "TimeSlab.h"
#include "TimeSlabSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
TimeSlabSolver::TimeSlabSolver(TimeSlab& timeslab)
  : ode(timeslab.ode), method(*timeslab.method), tol(0.0), maxiter(0),
    monitor(ode.get("ODE monitor convergence")),
    num_timeslabs(0), num_global_iterations(0), num_local_iterations(0),
    xnorm(0.0)
{
  // Choose tolerance
  chooseTolerance();

  // Get maximum number of iterations
  maxiter = ode.get("ODE maximum iterations");
}
//-----------------------------------------------------------------------------
TimeSlabSolver::~TimeSlabSolver()
{
  if ( num_timeslabs > 0 )
  {
    const double n = static_cast<double>(num_timeslabs);
    const double global_average = static_cast<double>(num_global_iterations) / n;
    const double local_average = static_cast<double>(num_local_iterations) / 
      static_cast<double>(num_global_iterations);
    message("Average number of global iterations per step: %.3f",
		global_average);
    message("Average number of local iterations per global iteration: %.3f",
		local_average);
  }

  message("Total number of (macro) time steps: %d", num_timeslabs);
}
//-----------------------------------------------------------------------------
bool TimeSlabSolver::solve()
{
  for (uint attempt = 0; attempt < maxiter; attempt++)
  {
    // Try to solve system
    if ( solve(attempt) )
      return true;
    
    // Check if we should try again
    if ( !retry() )
      return false;
  }

  return false;
}
//-----------------------------------------------------------------------------
bool TimeSlabSolver::solve(uint attempt)
{
  start();

  double d0 = 0.0;
  double d1 = 0.0;
  for (uint iter = 0; iter < maxiter; iter++)
  {
    // Do one iteration
    double d2 = iteration(tol, iter, d0, d1);

    // Use relative increment
    d2 /= xnorm + DOLFIN_EPS;
    
    // For debugging convergence
    if ( monitor )
      message("--- iter = %d: increment = %.3e", iter, d2);
    
    // Check convergenge
    if ( d2 < tol )
    {
      end();
      num_timeslabs += 1;
      num_global_iterations += iter + 1;
      if ( monitor )
	message("Time slab system of size %d converged in %d iterations.\n", size(), iter + 1);
      return true;
    }

    // Check divergence
    // FIXME: implement better check and make this a parameter
    if ( (iter > 0 && d2 > 1000.0 * d1) || !std::isnormal(d2) )
    {
      warning("Time slab system seems to be diverging.");
      return false;
    }
    
    d0 = d1;
    d1 = d2;
  }

  warning("Time slab system did not converge.");
  return false;
}
//-----------------------------------------------------------------------------
bool TimeSlabSolver::retry()
{
  // By default, we don't know how to make a new attempt
  return false;
}
//-----------------------------------------------------------------------------
void TimeSlabSolver::start()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void TimeSlabSolver::end()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void TimeSlabSolver::chooseTolerance()
{
  const double TOL   = ode.get("ODE tolerance");
  const double alpha = ode.get("ODE discrete tolerance factor");

  tol = ode.get("ODE discrete tolerance");
  if ( !ode.get("ODE fixed time step") )
    tol = std::min(tol, alpha*TOL);
  cout << "Using discrete tolerance tol = " << tol << "." << endl;
}
//-----------------------------------------------------------------------------
