// Copyright (C) 2003-2006 Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Benjamin Kehlet 2008
//
// First added:  2003
// Last changed: 2008-06-18

#include <dolfin/log/dolfin_log.h>
#include <dolfin/parameter/parameters.h>
#include "ODE.h"
#include "TimeStepper.h"
#include "ODESolver.h"
#include "ODESolution.h"
#include "Dual.h"

using namespace dolfin;

//-----------------------------------------------------------------------

void ODESolver::solve(ODE& ode, ODESolution& u) {
  begin("Solving ODE");  

  // Solve primal problem
  solvePrimal(ode, u);

  u.makeIndex();

  // Check if we should solve the dual problem  
  if ( ode.get("ODE solve dual problem") ) {
    solveDual(ode, u);
  }else {
    cout << "Not solving the dual problem as requested." << endl;
  }

  end();

}

//----------------------------------------------------------------------
void ODESolver::solve(ODE& ode)
{
  //create dummy object to hold the solution
  ODESolution u(ode);
  solve(ode, u);

}
//------------------------------------------------------------------------
void ODESolver::solvePrimal(ODE& ode, ODESolution& u)
{
  begin("Solving primal problem");
  
  // Solve primal problem
  TimeStepper::solve(ode, u);

  end();
}
//------------------------------------------------------------------------
void ODESolver::solveDual(ODE& ode, ODESolution& u)
{ 
  begin("Solving dual problem");

  // Create dual problem
  Dual dual(ode, u);

  dual.set("ODE solution file name", "solution_dual.py");
  dual.set("ODE save final solution", true);


  //create dummy objeect to hold the solution of the dual
  ODESolution u_dual(dual);

  // Solve dual problem
  TimeStepper::solve(dual, u_dual);

  end();
}
//------------------------------------------------------------------------


