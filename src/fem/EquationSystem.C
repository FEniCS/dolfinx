// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include "EquationSystem.hh"

//-----------------------------------------------------------------------------
EquationSystem::EquationSystem(int no_eq, int nsd) : Equation(nsd)
{
  this->no_eq = no_eq;
}
//-----------------------------------------------------------------------------
