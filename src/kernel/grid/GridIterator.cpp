// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Grid.h>
#include <dolfin/GridHierarchy.h>
#include <dolfin/GridIterator.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
GridIterator::GridIterator(const GridHierarchy& grids) : it(grids.grids)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
GridIterator::GridIterator
(const GridHierarchy& grids, Index index) : it(grids.grids, index)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
GridIterator::~GridIterator()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
GridIterator& GridIterator::operator++()
{
  ++it;
  return *this;
}
//-----------------------------------------------------------------------------
GridIterator& GridIterator::operator--()
{
  --it;
  return *this;
}
//-----------------------------------------------------------------------------
bool GridIterator::end()
{
  return it.end();
}
//-----------------------------------------------------------------------------
int GridIterator::index()
{
  return it.index();
}
//-----------------------------------------------------------------------------
GridIterator::operator GridPointer() const
{
  return *it;
}
//-----------------------------------------------------------------------------
Grid& GridIterator::operator*() const
{
  return **it;
}
//-----------------------------------------------------------------------------
Grid* GridIterator::operator->() const
{
  return *it;
}
//-----------------------------------------------------------------------------
