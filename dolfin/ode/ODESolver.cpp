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
#include "Dual.h"

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
  ODESolution u;
  u.dummy=true;
  solve(u);
}
//-----------------------------------------------------------------------
void ODESolver::solve(ODESolution& u)
{
  begin("Solving ODE over the time interval [0.0, %g]", to_double(ode.endtime()));

  // Start timing
  tic();

  // Solve primal problem
  solve_primal(u);

  // Check if we should solve the dual problem
  if (ode.parameters["solve_dual_problem"])
    solve_dual(u);
  else
    cout << "Not solving the dual problem as requested." << endl;

  // Report elapsed time
  info("ODE solution computed in %.3f seconds.", toc());

  end();
}
//------------------------------------------------------------------------
void ODESolver::solve_primal(ODESolution& u)
{
  begin("Solving primal problem");

  TimeStepper time_stepper(ode);
  time_stepper.solve(u);
  u.flush();
  end();
}
//------------------------------------------------------------------------
void ODESolver::solve_dual(ODESolution& u)
{
  begin("Solving dual problem");

  // Create dual problem
  Dual dual(ode, u);

  // Create dummy object to hold the solution of the dual
  ODESolution z;
  z.dummy=true;

  // Solve dual problem
  TimeStepper time_stepper(dual);
  time_stepper.solve(z);

  end();
}
//------------------------------------------------------------------------
