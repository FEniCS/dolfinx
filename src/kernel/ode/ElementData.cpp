// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_settings.h>
#include <dolfin/ElementBlock.h>
#include <dolfin/ElementData.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
ElementData::ElementData(int N) : N(N), current(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
ElementData::~ElementData()
{
  clear();
}
//-----------------------------------------------------------------------------
Element* ElementData::createElement(const Element::Type type, real t0, real t1,
				    int q, int index)
{
  // Create a new block if there are no blocks
  if ( blocks.empty() )
  {
    current = new ElementBlock(N);
    blocks.push_back(current);
  }

  return current->createElement(type, t0, t1, q, index);
}
//-----------------------------------------------------------------------------
Element* ElementData::element(unsigned int i, real t)
{
  return findpos(t)->element(i,t);
}
//-----------------------------------------------------------------------------
Element* ElementData::last(unsigned int i)
{
  return current->last(i);
}
//-----------------------------------------------------------------------------
void ElementData::clear()
{
  // Delete all blocks
  for (NewList<ElementBlock*>::iterator block = blocks.begin(); block != blocks.end(); ++block)
  {
    if ( *block )
      delete *block;
    *block = 0;
  }
  
  blocks.clear();
}
//-----------------------------------------------------------------------------
unsigned int ElementData::size() const
{
  return N;
}
//-----------------------------------------------------------------------------
ElementBlock* ElementData::findpos(real t)
{
  return blocks.back();
}
//-----------------------------------------------------------------------------
