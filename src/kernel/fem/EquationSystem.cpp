// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/EquationSystem.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
EquationSystem::EquationSystem(int dim, int noeq) : Equation(dim)
{
  this->noeq = noeq;
}
//-----------------------------------------------------------------------------
real EquationSystem::lhs(const ShapeFunction &u, const ShapeFunction &v)
{
  dolfin_error("Using EquationSystem for equation with only one component.");
  exit(1);
}
//-----------------------------------------------------------------------------
real EquationSystem::rhs(const ShapeFunction &v)
{
  dolfin_error("Using EquationSystem for equation with only one component.");
}
//-----------------------------------------------------------------------------
