// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/ODE.h>
#include <dolfin/Dual.h>
#include <dolfin/Function.h>
#include <dolfin/TimeStepper.h>
#include <dolfin/NewTimeStepper.h>
#include <dolfin/ODESolver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void ODESolver::solve(ODE& ode)
{
  Function u, phi;
  solve(ode, u, phi);
}
//-----------------------------------------------------------------------------
void ODESolver::solve(ODE& ode, Function& u)
{
  Function phi;
  solve(ode, u, phi);
}
//-----------------------------------------------------------------------------
void ODESolver::solve(ODE& ode, Function& u, Function& phi)
{
  // Check if we should solve the dual problem
  bool solve_dual = dolfin_get("solve dual problem");

  dolfin_start("Solving ODE");  

  // Solve primal problem
  solvePrimal(ode, u);

  // Solve dual problem
  if ( solve_dual )
    solveDual(ode, u, phi);
  else
    cout << "Not solving the dual problem as requested." << endl;

  cout << "Not computing an error estimate. " 
       << "The solution may be inaccurate." << endl;

  dolfin_end();
}
//-----------------------------------------------------------------------------
void ODESolver::solvePrimal(ODE& ode, Function& u)
{
  dolfin_start("Solving primal problem");
  
  // Initialize primal solution
  u.init(ode.size());
  u.rename("u", "primal");
  
  // Solve primal problem
  if ( dolfin_get("use new ode solver") )
    NewTimeStepper::solve(ode);
  else
    TimeStepper::solve(ode, u);

  dolfin_end();
}
//-----------------------------------------------------------------------------
void ODESolver::solveDual(ODE& ode, Function& u, Function& phi)
{
  dolfin_start("Solving dual problem");
  
  // Create dual problem
  Dual dual(ode, u);
  
  // Initialize dual solution phi
  phi.init(ode.size());
  phi.rename("phi", "dual");
  
  // Solve dual problem
  if ( dolfin_get("use new ode solver") )
    NewTimeStepper::solve(ode);
  else
    TimeStepper::solve(dual, phi);

  dolfin_end();
}
//-----------------------------------------------------------------------------
