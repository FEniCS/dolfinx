// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Solver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Solver::Solver() : grid(dummy_grid), ode(dummy_ode)
{

}
//-----------------------------------------------------------------------------
Solver::Solver(Grid& grid_) : grid(grid_), ode(dummy_ode)
{

}
//-----------------------------------------------------------------------------
Solver::Solver(ODE& ode_) : grid(dummy_grid), ode(ode_)
{

}
//-----------------------------------------------------------------------------
