#include <stdarg.h>

#include "dolfin_modules.h"
#include <dolfin/Problem.h>
#include <dolfin/Settings.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Problem::Problem(const char *problem)
{
  // Initialise the solver
  solver = dolfin_module_solver(problem);

  // Initialise settings
  settings = dolfin_module_settings(problem);
}
//-----------------------------------------------------------------------------
Problem::Problem(const char *problem, Grid &grid)
{
  // Initialise the solver
  solver = dolfin_module_solver(problem, grid);

  // Initialise settings
  settings = dolfin_module_settings(problem);
}
//-----------------------------------------------------------------------------
Problem::~Problem()
{
  if ( solver )
	 delete solver;
  solver = 0;

  if ( settings )
	 delete settings;
  settings = 0;
}
//-----------------------------------------------------------------------------
void Problem::set(const char *property, ...)
{
  va_list aptr;
  va_start(aptr, property);
  
  Settings::set_aptr(property, aptr);
  
  va_end(aptr);
}
//-----------------------------------------------------------------------------
void Problem::solve()
{
  solver->solve();
}
//-----------------------------------------------------------------------------
