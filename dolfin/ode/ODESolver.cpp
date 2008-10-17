// Copyright (C) 2003-2006 Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Benjamin Kehlet 2008
//
// First added:  2003
// Last changed: 2008-10-02

#include <dolfin/log/dolfin_log.h>
#include <dolfin/parameter/parameters.h>
#include "ODE.h"
#include "TimeStepper.h"
#include "ODESolver.h"
#include "ODESolution.h"
#include "Dual.h"

using namespace dolfin;

//-----------------------------------------------------------------------
void ODESolver::solve(ODE& ode, ODESolution& u)
{
  begin("Solving ODE");  

  // Solve primal problem
  solve_primal(ode, u);
  u.flush();

  // Check if we should solve the dual problem  
  if ( ode.get("ODE solve dual problem") )
    solve_dual(ode, u);
  else
    cout << "Not solving the dual problem as requested." << endl;

  end();
}
//----------------------------------------------------------------------
void ODESolver::solve(ODE& ode)
{
  // Create dummy object to hold the solution
  ODESolution u(ode);

  // Solve primal problem
  solve(ode, u);
}
//------------------------------------------------------------------------
void ODESolver::solve_primal(ODE& ode, ODESolution& u)
{
  begin("Solving primal problem");
  
  // Solve primal problem
  TimeStepper::solve(ode, u);

  end();
}
//------------------------------------------------------------------------
void ODESolver::solve_dual(ODE& ode, ODESolution& u)
{ 
  begin("Solving dual problem");

  #ifdef HAS_GMP
    error("Solving dual with extended precision not implemented");
  #endif

  // Create dual problem
  Dual dual(ode, u);
  dual.set("ODE solution file name", "solution_dual.py");
  dual.set("ODE save final solution", true);

  // Create dummy objeect to hold the solution of the dual
  ODESolution u_dual(dual);

  // Solve dual problem
  TimeStepper::solve(dual, u_dual);

  end();
}
//------------------------------------------------------------------------
