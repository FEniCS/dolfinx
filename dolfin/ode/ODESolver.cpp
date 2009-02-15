// Copyright (C) 2003-2009 Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Benjamin Kehlet 2008
//
// First added:  2003
// Last changed: 2009-02-09

#include <dolfin/log/dolfin_log.h>
#include <dolfin/parameter/parameters.h>
#include <dolfin/common/timing.h>
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
  ODESolution u(ode);
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
  u.flush();

  // Check if we should solve the dual problem  
  if (ode.get("ODE solve dual problem"))
    solve_dual(u);
  else
    cout << "Not solving the dual problem as requested." << endl;

  // Report elapsed time
  message("ODE solution computed in %.3f seconds.", toc());

  end();
}
//------------------------------------------------------------------------
void ODESolver::solve_primal(ODESolution& u)
{
  begin("Solving primal problem");

  TimeStepper time_stepper(ode);
  time_stepper.solve(u);

  end();
}
//------------------------------------------------------------------------
void ODESolver::solve_dual(ODESolution& u)
{ 
  begin("Solving dual problem");

  // Create dual problem
  Dual dual(ode, u);

  if (dolfin_changed("floating-point precision")) 
  {
    warning("Solving dual with extended precision, not supported. Using double precision.");
    // Set discrete tolerance to default value.
    dual.set("ODE discrete tolerance", DOLFIN_SQRT_EPS);
  }


  dual.set("ODE solution file name", "solution_dual.py");
  dual.set("ODE save final solution", true);

  // Create dummy object to hold the solution of the dual
  ODESolution z(dual);

  // Solve dual problem
  TimeStepper time_stepper(dual);
  time_stepper.solve(z);

  end();
}
//------------------------------------------------------------------------
