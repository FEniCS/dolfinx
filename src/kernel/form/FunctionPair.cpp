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
FunctionPair::FunctionPair(NewArray<real>& w, Function& f)
{
  this->w = &w;
  this->f = &f;
}
//-----------------------------------------------------------------------------
FunctionPair::~FunctionPair()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void FunctionPair::update(const Cell& cell, const NewPDE& pde)
{
  // Update local values from global values
  f->update(*w, cell, pde);
}
//-----------------------------------------------------------------------------
