// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/ElementGroupList.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
ElementGroupList::ElementGroupList(TimeSlab& timeslab) : groups(0)
{
  // Compute the size of the list
  unsigned int size = 0;
  timeslab.countElementGroups(size);
  
  // Create the list
  groups = new NewArray<ElementGroup*>(size);

  // Add element groups
  unsigned int pos = 0;
  timeslab.addElementGroups(*groups, pos);
}
//-----------------------------------------------------------------------------
ElementGroupList::~ElementGroupList()
{
  // Delete the list
  if ( groups )
    delete groups;
  groups = 0;
}
//-----------------------------------------------------------------------------
unsigned int ElementGroupList::size() const
{
  dolfin_assert(groups);
  return groups->size();
}
//-----------------------------------------------------------------------------
