// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/TimeSlab.h>
#include <dolfin/ElementGroup.h>
#include <dolfin/ElementIterator.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
ElementIterator::ElementIterator(TimeSlab& timeslab)
{
  it = new TimeSlabElementIterator(timeslab);
}
//-----------------------------------------------------------------------------
ElementIterator::ElementIterator(ElementGroup& group)
{
  it = new ElementGroupElementIterator(group);
}
//-----------------------------------------------------------------------------
ElementIterator::~ElementIterator()
{
  if ( it )
    delete it;
  it = 0;
}
//-----------------------------------------------------------------------------
ElementIterator::operator ElementPointer() const
{
  dolfin_assert(it);
  return it->pointer();
}
//-----------------------------------------------------------------------------
ElementIterator& ElementIterator::operator++()
{
  dolfin_assert(it);
  ++(*it);
  return *this;
}
//-----------------------------------------------------------------------------
Element& ElementIterator::operator*() const
{
  dolfin_assert(it);
  return *(*it);
}
//-----------------------------------------------------------------------------
Element* ElementIterator::operator->() const
{
  dolfin_assert(it);
  return it->pointer();
}
//-----------------------------------------------------------------------------
bool ElementIterator::end()
{
  dolfin_assert(it);
  return it->end();
}
//-----------------------------------------------------------------------------
// ElementIterator::TimeSlabElementIterator
//-----------------------------------------------------------------------------
ElementIterator::TimeSlabElementIterator::
TimeSlabElementIterator(TimeSlab& timeslab)
{
  

}
//-----------------------------------------------------------------------------
void ElementIterator::TimeSlabElementIterator::operator++()
{


}
//-----------------------------------------------------------------------------
Element& ElementIterator::TimeSlabElementIterator::operator*() const
{
  return *element;  
}
//-----------------------------------------------------------------------------
Element* ElementIterator::TimeSlabElementIterator::operator->() const
{
  return element;
}
//-----------------------------------------------------------------------------
Element* ElementIterator::TimeSlabElementIterator::pointer() const
{
  return element;
}
//-----------------------------------------------------------------------------
bool ElementIterator::TimeSlabElementIterator::end()
{
  return true;
}
//-----------------------------------------------------------------------------
// ElementIterator::ElementGroupIterator
//-----------------------------------------------------------------------------
ElementIterator::ElementGroupElementIterator::
ElementGroupElementIterator(ElementGroup& group)
{
  it = group.elements.begin();
  at_end = group.elements.end();
}
//-----------------------------------------------------------------------------
void ElementIterator::ElementGroupElementIterator::operator++()
{
  ++it;
}
//-----------------------------------------------------------------------------
Element& ElementIterator::ElementGroupElementIterator::operator*() const
{
  return **it;
}
//-----------------------------------------------------------------------------
Element* ElementIterator::ElementGroupElementIterator::operator->() const
{
  return *it;
}
//-----------------------------------------------------------------------------
Element* ElementIterator::ElementGroupElementIterator::pointer() const
{
  return *it;
}
//-----------------------------------------------------------------------------
bool ElementIterator::ElementGroupElementIterator::end()
{
  return it == at_end;
}
//-----------------------------------------------------------------------------
