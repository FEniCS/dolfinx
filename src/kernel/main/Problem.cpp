#include <dolfin/Problem.h>
#include "dolfin_modules.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Problem::Problem(const char *problem)
{
  solver = dolfin_module_solver(problem);
  grid = 0;
}
//-----------------------------------------------------------------------------
Problem::Problem(const char *problem, Grid *grid)
{
  solver = dolfin_module_solver(problem,grid);
  this->grid = grid;
}
//-----------------------------------------------------------------------------
void Problem::set(const char *property, ...)
{
  
}
//-----------------------------------------------------------------------------
void Problem::solve()
{
  solver->solve();
}
//-----------------------------------------------------------------------------
