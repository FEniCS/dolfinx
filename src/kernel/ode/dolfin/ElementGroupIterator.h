// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ELEMENT_GROUP_ITERATOR_H
#define __ELEMENT_GROUP_ITERATOR_H

namespace dolfin
{

  class ElementGroup;
  class ElementGrouList;

  typedef ElementGroup* ElementGroupPointer;

  /// Iterator for access to element groups stored in time slabs.

  class ElementGroupIterator
  {
  public:

    /// Constructor
    ElementGroupIterator(ElementGroupList& groups);
    
    /// Destructor
    ~ElementGroupIterator();
    
    /// Cast to element group pointer
    operator ElementGroupPointer() const;
    
    /// Operator ++
    ElementGroupIterator& operator++();
    
    /// Operator *
    ElementGroup& operator*() const;

    /// Operator ->
    ElementGroup* operator->() const;

    /// Check if iterator has reached the end
    bool end();

  private:

    NewArray<ElementGroup*>::iterator it;
    NewArray<ElementGroup*>::iterator at_end;

  };

}

#endif
