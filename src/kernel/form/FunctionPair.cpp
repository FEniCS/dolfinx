// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Function.h>
#include <dolfin/FunctionPair.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
FunctionPair::FunctionPair() : w(0), f(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FunctionPair::~FunctionPair()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
