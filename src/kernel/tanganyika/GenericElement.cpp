// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/GenericElement.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericElement::GenericElement(int q)
{
  dolfin_assert(q >= 0);
  
  this->q = q;
  values = new real[q+1];
}
//-----------------------------------------------------------------------------
GenericElement::~GenericElement()
{
  delete [] values;
}
//-----------------------------------------------------------------------------
