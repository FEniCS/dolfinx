// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/GenericFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericFunction::GenericFunction(int dim, int size)
{
  this->dim = dim;
  this->size = size;
}
//-----------------------------------------------------------------------------
