// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/PDESystem.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
PDESystem::PDESystem(int dim, int noeq) : PDE(dim)
{
  this->noeq = noeq;
}
//-----------------------------------------------------------------------------
real PDESystem::lhs(const ShapeFunction &u, const ShapeFunction &v)
{
  dolfin_error("Using PDESystem for equation with only one component.");
  exit(1);
}
//-----------------------------------------------------------------------------
real PDESystem::rhs(const ShapeFunction &v)
{
  dolfin_error("Using PDESystem for equation with only one component.");
}
//-----------------------------------------------------------------------------
