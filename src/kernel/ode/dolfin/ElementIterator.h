// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ELEMENT_ITERATOR_H
#define __ELEMENT_ITERATOR_H

#include <dolfin/NewArray.h>

namespace dolfin
{

  class Element;
  class ElementGroup;
  class ElementGroupList;

  typedef Element* ElementPointer;

  /// Iterator for access to elements stored in element group lists
  /// (time slabs) or element groups.

  class ElementIterator
  {
  public:

    /// Constructor, create element iterator for given element group list
    ElementIterator(ElementGroupList& groups);

    /// Constructor, create element iterator for given element group
    ElementIterator(ElementGroup& group);
    
    /// Destructor
    ~ElementIterator();
    
    /// Cast to element pointer
    operator ElementPointer() const;
    
    /// Operator ++
    ElementIterator& operator++();
    
    /// Operator *
    Element& operator*() const;

    /// Operator ->
    Element* operator->() const;

    /// Check if iterator has reached the end
    bool end();
    
    /// Base class for element iterators
    class GenericElementIterator
    {
    public:
      virtual void operator++() = 0;
      virtual Element& operator*() const = 0;
      virtual Element* operator->() const = 0;
      virtual Element* pointer() const = 0;
      virtual bool end() = 0;
    };

    /// Iterator for time element group lists
    class ElementGroupListElementIterator : public GenericElementIterator
    {
    public:
      ElementGroupListElementIterator(ElementGroupList& groups);
      void operator++();
      Element& operator*() const;
      Element* operator->() const;
      Element* pointer() const;
      bool end();
    private:
      Element* element;
      NewArray<Element*>::iterator element_it;
      NewArray<Element*>::iterator element_at_end;
      NewArray<ElementGroup*>::iterator group_it;
      NewArray<ElementGroup*>::iterator group_at_end;
    };

    /// Iterator for element group elements
    class ElementGroupElementIterator : public GenericElementIterator
    {
    public:
      ElementGroupElementIterator(ElementGroup& group);
      void operator++();
      Element& operator*() const;
      Element* operator->() const;
      Element* pointer() const;
      bool end();
    private:
      NewArray<Element*>::iterator it;
      NewArray<Element*>::iterator at_end;
    };

  private:

    GenericElementIterator* it;

  };

}

#endif
