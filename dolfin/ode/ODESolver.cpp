// Copyright (C) 2003-2009 Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Benjamin Kehlet 2008
//
// First added:  2003
// Last changed: 2009-09-08

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
  tic();

  // Solve primal problem
  TimeStepper time_stepper(ode);
  time_stepper.solve();

  // Report elapsed time
  info(PROGRESS, "ODE solution computed in %.3f seconds.", toc());

  end();
}
//-----------------------------------------------------------------------
void ODESolver::solve(ODESolution& u)
{
  begin("Solving ODE over the time interval [0.0, %g]", to_double(ode.endtime()));

  // Start timing
  tic();

  // Solve primal problem
  TimeStepper time_stepper(ode, u);
  time_stepper.solve();
  u.flush();

  // Report elapsed time
  info(PROGRESS, "ODE solution computed in %.3f seconds.", toc());

  end();
}
//------------------------------------------------------------------------

