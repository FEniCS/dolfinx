// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_settings.h>
#include <dolfin/TimeSlabSolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
TimeSlabSolver::TimeSlabSolver(NewTimeSlab& timeslab, const NewMethod& method)
  : ts(timeslab), method(method), tol(0.0), maxiter(0)
{
  // Get tolerance
  tol = dolfin_get("tolerance");
  tol *= 0.01;
  cout << "Using tolerance tol = " << tol << endl;

  // Get maximum number of iterations
  maxiter = dolfin_get("maximum iterations");
}
//-----------------------------------------------------------------------------
TimeSlabSolver::~TimeSlabSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void TimeSlabSolver::solve()
{
  for (uint iter = 0; iter < maxiter; iter++)
  {
    // Do one iteration
    real increment = iteration();
    
    cout << "--- increment = " << increment << " ---" << endl;

    // Check convergenge
    if ( increment < tol )
      return;
  }

  dolfin_error("Time slab system did not converge.");
}
//-----------------------------------------------------------------------------
