// Copyright (C) 2005-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-01-05
// Last changed: 2009-09-08

#include <cmath>
#include <dolfin/common/constants.h>
#include <dolfin/log/dolfin_log.h>
#include "TimeSlab.h"
#include "TimeSlabSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
TimeSlabSolver::TimeSlabSolver(TimeSlab& timeslab)
  : ode(timeslab.ode), method(*timeslab.method), tol(0.0), maxiter(0),
    monitor(ode.parameters["monitor_convergence"]),
    num_timeslabs(0), num_global_iterations(0), num_local_iterations(0),
    xnorm(0.0)
{
  // Choose tolerance
  choose_tolerance();

  // Get maximum number of iterations
  maxiter = ode.parameters["maximum_iterations"];
}
//-----------------------------------------------------------------------------
TimeSlabSolver::~TimeSlabSolver()
{
  if (num_timeslabs > 0)
  {
    const real n = static_cast<real>(num_timeslabs);
    const real global_average = static_cast<real>(num_global_iterations) / n;
    const real local_average = static_cast<real>(num_local_iterations) /
      static_cast<real>(num_global_iterations);
    info("Average number of global iterations per step: %.3f",
	    to_double(global_average));
    info("Average number of local iterations per global iteration: %.3f",
	    to_double(local_average));
  }

  info("Total number of (macro) time steps: %d", num_timeslabs);
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

  real d0 = 0.0;
  real d1 = 0.0;
  for (uint iter = 0; iter < maxiter; iter++)
  {
    // Do one iteration
    real d2 = iteration(tol, iter, d0, d1);

    // Use relative increment
    d2 /= xnorm + real_epsilon();

    // For debugging convergence
    if (monitor)
      info("--- iter = %d: increment = %.3e", iter, to_double(d2));

    // Check convergenge
    if (d2 < tol)
    {
      end();
      num_timeslabs += 1;
      num_global_iterations += iter + 1;
      if (monitor)
	info("Time slab system of size %d converged in %d iterations.\n", size(), iter + 1);
      return true;
    }

    // Check divergence
    // FIXME: implement better check and make this a parameter
    // Note that the last check is skipped when working extended precision
    // as GMP has no notion of NaN or inifinity
    if ((iter > 0 && d2 > 1000.0 * d1) || !isnormal(d2))
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
void TimeSlabSolver::choose_tolerance()
{
  tol = ode.parameters["discrete_tolerance"].get_real();

  if (!ode.parameters["fixed_time_step"])
  {
    const real TOL = ode.parameters["tolerance"].get_real();
    const real alpha = ode.parameters["discrete_tolerance_factor"].get_real();
    tol = real_min(tol, alpha*TOL);
  }
  cout << "Using discrete tolerance tol = " << tol << "." << endl;
}
//-----------------------------------------------------------------------------
