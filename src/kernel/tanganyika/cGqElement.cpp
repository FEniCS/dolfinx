// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/cGqElement.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void cGqElement::update()
{
  for (int i = 0; i <= q; i++)
    values[i] = 0.0;
}
//-----------------------------------------------------------------------------
