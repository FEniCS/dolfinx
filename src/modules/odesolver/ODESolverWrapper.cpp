// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include "ODESolverWrapper.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ODESolverWrapper::ODESolverWrapper(ODE& ode) : Solver(ode)
{
  dolfin_parameter(Parameter::REAL, "final time", 0.0);
}
//-----------------------------------------------------------------------------
const char* ODESolverWrapper::description()
{
  return "General ODE";
}
//-----------------------------------------------------------------------------
void ODESolverWrapper::solve()
{
  // Get final time
  real T = dolfin_get("final time");
  
  // Solve ODE
  solver.solve(ode, T);
}
//-----------------------------------------------------------------------------
