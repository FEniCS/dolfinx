// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/cGqElement.h>
#include <dolfin/dGqElement.h>
#include <dolfin/Element.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Element::Element(int q)
{
  element = new cGqElement(q);
}
//-----------------------------------------------------------------------------
Element::~Element()
{
  delete element;
}
//-----------------------------------------------------------------------------
