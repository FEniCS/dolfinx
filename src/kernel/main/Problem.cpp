// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <stdarg.h>

#include "dolfin_modules.h"
#include <dolfin/Problem.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Problem::Problem(const char *problem)
{
  // Initialise the solver
  solver = dolfin_module_solver(problem);
}
//-----------------------------------------------------------------------------
Problem::Problem(const char *problem, Grid &grid)
{
  // Initialise the solver
  solver = dolfin_module_solver(problem, grid);
}
//-----------------------------------------------------------------------------
Problem::~Problem()
{
  if ( solver )
    delete solver;
  solver = 0;
}
//-----------------------------------------------------------------------------
void Problem::set(const char *property, ...)
{
  va_list aptr;
  va_start(aptr, property);
  
  // Settings are global, although the syntax problem.set() seems to indicate
  // that parameters are local to a given problem.
  dolfin_set_aptr(property, aptr);
  
  va_end(aptr);
}
//-----------------------------------------------------------------------------
void Problem::solve()
{
  solver->solve();
}
//-----------------------------------------------------------------------------
