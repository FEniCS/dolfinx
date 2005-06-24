// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_settings.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/TimeSlabSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
TimeSlabSolver::TimeSlabSolver(TimeSlab& timeslab)
  : ode(timeslab.ode), method(*timeslab.method), tol(0.0), maxiter(0),
    monitor(dolfin_get("monitor convergence")),
    num_timeslabs(0), num_iterations(0)
{
  // Get tolerance
  const real TOL = dolfin_get("tolerance");
  if ( dolfin_parameter_changed("discrete tolerance") )
  {
    tol = dolfin_get("discrete tolerance");
  }
  else
  {
    const real alpha = dolfin_get("discrete tolerance factor");
    tol = alpha * TOL;
  }

  cout << "Using discrete tolerance tol = " << tol << "." << endl;

  // Get maximum number of iterations
  maxiter = dolfin_get("maximum iterations");
}
//-----------------------------------------------------------------------------
TimeSlabSolver::~TimeSlabSolver()
{
  if ( num_timeslabs > 0 )
  {
    const real n = static_cast<real>(num_iterations) / static_cast<real>(num_timeslabs);
    dolfin_info("Average number of global iterations per step: %.2f.", n);
  }

  dolfin_info("Total number of (macro) time steps: %d.", num_timeslabs);
}
//-----------------------------------------------------------------------------
bool TimeSlabSolver::solve()
{
  start();

  real d0 = 0.0;
  for (uint iter = 0; iter < maxiter; iter++)
  {
    // Do one iteration
    real d1 = iteration(iter, tol);
    if ( monitor )
      dolfin_info("--- iter = %d: increment = %.3e", iter, d1);
    
    // Check convergenge
    if ( d1 < tol )
    {
      end();
      num_timeslabs += 1;
      num_iterations += iter + 1;
      if ( monitor )
	dolfin_info("Time slab system of size %d converged in %d iterations.", size(), iter + 1);
      return true;
    }

    // Check divergence
    // FIXME: implement better check and make this a parameter
    if ( iter > 0 && d1 > 1000.0 * d0 )
    {
      dolfin_warning("Time slab system seems to be diverging, solution stopped.");
      return false;
    }
    
    d0 = d1;
  }

  dolfin_warning("Time slab system did not converge, solution stopped.");
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
