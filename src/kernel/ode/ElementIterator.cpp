// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/ElementGroup.h>
#include <dolfin/ElementGroupList.h>
#include <dolfin/ElementIterator.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
ElementIterator::ElementIterator(ElementGroupList& groups)
{
  it = new ElementGroupListElementIterator(groups);
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
// ElementIterator::ElementGroupListElementIterator
//-----------------------------------------------------------------------------
ElementIterator::ElementGroupListElementIterator::
ElementGroupListElementIterator(ElementGroupList& groups)
{
  // Initialize iterators for element groups
  group_it = groups.groups->begin();
  group_at_end = groups.groups->end();

  // Initialize iterators for elements
  element_it = (*group_it)->elements.begin();
  element_at_end = (*group_it)->elements.end();
}
//-----------------------------------------------------------------------------
void ElementIterator::ElementGroupListElementIterator::operator++()
{
  // Step element iterator
  ++element_it;
  
  // Check if we need to step the element group iterator
  if ( element_it == element_at_end )
  {
    cout << "Reached end of group, stepping to next group" << endl;

    ++group_it;
    if ( group_it != group_at_end )
    {
      cout << "Found next group" << endl;
      element_it = (*group_it)->elements.begin();
      element_at_end = (*group_it)->elements.end();
    }
  }
}
//-----------------------------------------------------------------------------
Element& ElementIterator::ElementGroupListElementIterator::operator*() const
{
  return **element_it;
}
//-----------------------------------------------------------------------------
Element* ElementIterator::ElementGroupListElementIterator::operator->() const
{
  return *element_it;
}
//-----------------------------------------------------------------------------
Element* ElementIterator::ElementGroupListElementIterator::pointer() const
{
  return *element_it;
}
//-----------------------------------------------------------------------------
bool ElementIterator::ElementGroupListElementIterator::end()
{
  return element_it == element_at_end && group_it == group_at_end;
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
