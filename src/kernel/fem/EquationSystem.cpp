// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <iostream>
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
  // FIXME: Use logging system
  cout << "Error: Using EquationSystem for equation with only one component." << endl;
  exit(1);
}
//-----------------------------------------------------------------------------
real EquationSystem::rhs(const ShapeFunction &v)
{
  // FIXME: Use logging system
  cout << "Error: Using EquationSystem for equation with only one component." << endl;
  exit(1);
}
//-----------------------------------------------------------------------------
