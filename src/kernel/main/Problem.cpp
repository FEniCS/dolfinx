#include <stdarg.h>

#include "dolfin_modules.h"
#include <dolfin/Problem.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Problem::Problem(const char *problem)
{
  // Initialise the solver
  solver = dolfin_module_solver(problem);

  // Initialise settings
  dolfin_module_init_settings(problem);
}
//-----------------------------------------------------------------------------
Problem::Problem(const char *problem, Grid &grid)
{
  // Initialise the solver
  solver = dolfin_module_solver(problem, grid);

  // Initialise settings
  dolfin_module_init_settings(problem);
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
  
  Settings::set_aptr(property, aptr);
  
  va_end(aptr);
}
//-----------------------------------------------------------------------------
void Problem::solve()
{
  solver->solve();
}
//-----------------------------------------------------------------------------
