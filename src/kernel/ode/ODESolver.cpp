// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/ODE.h>
#include <dolfin/Dual.h>
#include <dolfin/Function.h>
#include <dolfin/TimeStepper.h>
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
  dolfin_start("Solving ODE");
  
  // Get size of system
  unsigned int N = ode.size();

  // Rename primal and dual solutions
  u.rename("u", "primal");
  phi.rename("phi", "dual");

  dolfin_start("Solving primal problem");

  // Initialize primal solution
  u.init(N);
  
  // Solve primal problem
  TimeStepper::solve(ode, u);

  dolfin_end();

  dolfin_start("Solving dual problem");

  // Create dual problem
  Dual dual(ode, u);

  // Initialize dual solution phi
  phi.init(N);
  
  // Solve dual problem
  TimeStepper::solve(dual, phi);

  dolfin_end();

  dolfin_end();
}
//-----------------------------------------------------------------------------
