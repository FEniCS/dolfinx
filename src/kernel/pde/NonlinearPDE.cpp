// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2006-2007.
//
// First added:  2005-10-24
// Last changed: 2007-04-19

#include <dolfin/NonlinearPDE.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NonlinearPDE::NonlinearPDE(Form& a,
                           Form& L,
                           Mesh& mesh,
                           Array<BoundaryCondition*> bcs)
  : GenericPDE(a, L, mesh, bcs)
{
  dolfin_info("Creating linear PDE with %d boundary condition(s).", bcs.size());
}
//-----------------------------------------------------------------------------
NonlinearPDE::~NonlinearPDE()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NonlinearPDE::solve(Function& u)
{
  dolfin_begin("Solving nonlinear PDE.");

  dolfin_error("Not implemented.");
}
//-----------------------------------------------------------------------------
