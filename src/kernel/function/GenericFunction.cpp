// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-11-28
// Last changed: 2005-11-28

#include <dolfin/GenericFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericFunction::GenericFunction()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
GenericFunction::~GenericFunction()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void GenericFunction::interpolate(Function& fsource)
{
  dolfin_error("Cannot interpolate to this type of function.");
}
//-----------------------------------------------------------------------------
