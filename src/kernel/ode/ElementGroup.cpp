// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/ElementGroup.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
ElementGroup::ElementGroup()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
ElementGroup::~ElementGroup()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void ElementGroup::add(Element& element)
{
  elements.push_back(&element);
}
//-----------------------------------------------------------------------------
unsigned int ElementGroup::size() const
{
  return elements.size();
}
//-----------------------------------------------------------------------------
