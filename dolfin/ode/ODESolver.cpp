// Copyright (C) 2003-2009 Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Benjamin Kehlet 2008
//
// First added:  2003
// Last changed: 2011-03-17

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/timing.h>
#include <dolfin/parameter/GlobalParameters.h>
#include "ODE.h"
#include "TimeStepper.h"
#include "ODESolver.h"
#include "ODESolution.h"

using namespace dolfin;

//------------------------------------------------------------------------
ODESolver::ODESolver(ODE& ode) : ode(ode)
{
  // Do nothing
}
//------------------------------------------------------------------------
ODESolver::~ODESolver()
{
  // Do nothing
}
//------------------------------------------------------------------------
void ODESolver::solve()
{
  begin("Solving ODE over the time interval [0.0, %g]", to_double(ode.endtime()));

  // Start timing
  double tt = time();

  // Solve primal problem
  TimeStepper time_stepper(ode);
  time_stepper.solve();

  // Report elapsed time
  tt = time() - tt;
  log(PROGRESS, "ODE solution computed in %.3f seconds.", tt);

  end();
}
//-----------------------------------------------------------------------
void ODESolver::solve(ODESolution& u)
{
  begin("Solving ODE over the time interval [0.0, %g]", to_double(ode.endtime()));

  // Start timing
  double tt = time();

  // Solve primal problem
  TimeStepper time_stepper(ode, u);
  time_stepper.solve();
  u.flush();

  // Report elapsed time
  tt = time() - tt;
  log(PROGRESS, "ODE solution computed in %.3f seconds.", tt);

  end();
}
//------------------------------------------------------------------------
