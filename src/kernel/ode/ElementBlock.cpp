// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/ElementBlock.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
ElementBlock::ElementBlock(int N) : components(N)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
ElementBlock::~ElementBlock()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Element* ElementBlock::createElement(Element::Type type, real t0, real t1,
				     int q, int index)
{
  return components[index].createElement(type, t0, t1, q, index);
}
//-----------------------------------------------------------------------------
Element* ElementBlock::element(unsigned int index, real t)
{
  return components[index].element(t);
}
//-----------------------------------------------------------------------------
Element* ElementBlock::last(unsigned int i)
{
  dolfin_assert(i < components.size());
  return components[i].last();
}
//-----------------------------------------------------------------------------
unsigned int ElementBlock::size() const
{
  return components.size();
}
//-----------------------------------------------------------------------------
