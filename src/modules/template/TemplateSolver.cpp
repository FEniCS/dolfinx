// Copyright (C) 2002 [fill in name]
// Licensed under the GNU GPL Version 2.

#include "TemplateSolver.h"
#include "Template.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
TemplateSolver::TemplateSolver(Grid& grid) : Solver(grid)
{
  dolfin_parameter(Parameter::REAL, "my parameter", 42.0);
}
//-----------------------------------------------------------------------------
const char *TemplateSolver::description()
{
  return "My new equation";
}
//-----------------------------------------------------------------------------
void TemplateSolver::solve()
{
  cout << "Solving..." << endl;
}
//-----------------------------------------------------------------------------
