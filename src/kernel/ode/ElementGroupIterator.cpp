// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/ElementGroupList.h>
#include <dolfin/ElementGroupIterator.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
ElementGroupIterator::ElementGroupIterator(TimeSlab& timeslab)
{
  it = timeslab.
  // Do nothing
}
//-----------------------------------------------------------------------------
ElementGroupIterator::~ElementGroupIterator()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
ElementGroupIterator::operator ElementGroupPointer() const
{
  dolfin_assert(group);
  return group;
}
//-----------------------------------------------------------------------------
ElementGroupIterator& ElementGroupIterator::operator++()
{
  if ( at_end )
    return *this;

  return *this;
}
//-----------------------------------------------------------------------------
ElementGroup& ElementGroupIterator::operator*() const
{
  dolfin_assert(group);
  return *group;
}
//-----------------------------------------------------------------------------
ElementGroup* ElementGroupIterator::operator->() const
{
  dolfin_assert(group);
  return group;
}
//-----------------------------------------------------------------------------
bool ElementGroupIterator::end()
{
  return at_end;
}
//-----------------------------------------------------------------------------
