// Copyright (C) 2002 [fill in name]
// Licensed under the GNU GPL Version 2.

#include "TemplateSolver.h"
#include "Template.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
const char *TemplateSolver::description()
{
  return "My new equation";
}
//-----------------------------------------------------------------------------
void TemplateSolver::solve()
{
  std::cout << "Solving..." << std::endl;
}
//-----------------------------------------------------------------------------
