// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __ELEMENT_ITERATOR_H
#define __ELEMENT_ITERATOR_H

#include <dolfin/NewArray.h>

namespace dolfin
{

  class Element;
  class TimeSlab;
  class ElementGroup;

  typedef Element* ElementPointer;

  /// Iterator for access to elements stored in time slabs or
  /// element groups.

  class ElementIterator
  {
  public:

    /// Constructor, create element iterator for given time slab
    ElementIterator(TimeSlab& timeslab);

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

    /// Iterator for time slab elements
    class TimeSlabElementIterator : public GenericElementIterator
    {
    public:
      TimeSlabElementIterator(TimeSlab& timeslab);
      void operator++();
      Element& operator*() const;
      Element* operator->() const;
      Element* pointer() const;
      bool end();
    private:
      Element* element;
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
