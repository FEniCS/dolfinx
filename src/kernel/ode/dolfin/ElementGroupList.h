// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ELEMENT_GROUP_LIST_H
#define __ELEMENT_GROUP_LIST_H

#include <dolfin/NewArray.h>

namespace dolfin
{

  class TimeSlab;
  class ElementGroup;

  /// ElementGroupList represents a list of all element groups contained
  /// in a given time slab. This is used to iterate over element groups
  /// in the fixed point iteration.

  class ElementGroupList
  {
  public:

    /// Constructor, create the element group list from a given time slab
    ElementGroupList(TimeSlab& timeslab);

    /// Destructor
    ~ElementGroupList();

  private:

    NewArray<ElementGroup*>* groups;

  };

}

#endif
