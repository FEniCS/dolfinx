// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ELEMENT_GROUP_H
#define __ELEMENT_GROUP_H

#include <dolfin/ElementIterator.h>
#include <dolfin/NewArray.h>

namespace dolfin
{

  class Element;

  /// ElementGroup represents a group (list) of elements
  /// an is the basic building block of time slabs.

  class ElementGroup
  {
  public:
    
    /// Constructor
    ElementGroup();

    /// Destructor
    ~ElementGroup();

    /// Add a new element
    void add(Element& element);

    /// Return size of element group (number of elements)
    unsigned int size() const;
    
    // Friends
    friend class ElementIterator::ElementGroupElementIterator;

  private:

    // The list of elements
    NewArray<Element*> elements;
    
  };

}

#endif
