// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/ElementGroupList.h>
#include <dolfin/ElementGroupIterator.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
ElementGroupIterator::ElementGroupIterator(ElementGroupList& groups)
{
  it = groups.groups->begin();
  at_end = groups.groups->end();
}
//-----------------------------------------------------------------------------
ElementGroupIterator::~ElementGroupIterator()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
ElementGroupIterator::operator ElementGroupPointer() const
{
  return *it;
}
//-----------------------------------------------------------------------------
ElementGroupIterator& ElementGroupIterator::operator++()
{
  ++it;
  return *this;
}
//-----------------------------------------------------------------------------
ElementGroup& ElementGroupIterator::operator*() const
{
  return **it;
}
//-----------------------------------------------------------------------------
ElementGroup* ElementGroupIterator::operator->() const
{
  return *it;
}
//-----------------------------------------------------------------------------
bool ElementGroupIterator::end()
{
  return it == at_end;
}
//-----------------------------------------------------------------------------
