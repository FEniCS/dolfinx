// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/TableSparsity.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
TableSparsity::TableSparsity(int N) : GenericSparsity(N)
{
  list = new ShortList<int>[N];
}
//-----------------------------------------------------------------------------
TableSparsity::~TableSparsity()
{
  if ( list )
    delete list;
  list = 0;
}
//-----------------------------------------------------------------------------
void TableSparsity::setsize(int i, int size)
{
  dolfin_assert(i >= 0);
  dolfin_assert(i < N);

  if ( size > N )
    dolfin_error("Number of dependencies cannot be larger than system size.");

  list[i].init(size);

  // Index -1 denotes an empty position
  for (int pos = 0; pos < size; pos++)
    list[i](pos) = -1;
}
//-----------------------------------------------------------------------------
void TableSparsity::set(int i, int j)
{
  dolfin_assert(i >= 0);
  dolfin_assert(i < N);

  // Find first empty position
  for (int pos = 0; pos < list[i].size(); pos++) {
    if ( list[i](pos) == j )
      return;
    else if ( list[i](pos) == -1 ) {
      list[i](pos) = j;
      return;
    }
  }

  dolfin_error1("All dependencies for component %d alreay specified.", i);
}
//-----------------------------------------------------------------------------
GenericSparsity::Type TableSparsity::type() const
{
  return table;
}
//-----------------------------------------------------------------------------
TableSparsity::Iterator* TableSparsity::createIterator(int i) const
{
  dolfin_assert(i >= 0);
  dolfin_assert(i < N);

  return new Iterator(i, *this);
}
//-----------------------------------------------------------------------------
TableSparsity::Iterator::Iterator(int i, const TableSparsity& sparsity)
  : GenericSparsity::Iterator(i), s(sparsity)
{
  pos = 0;

  if ( s.list[i].size() == 0 )
    at_end = true;
  else if ( s.list[i](0) == -1 )
    at_end = true;
  else
    at_end = false;
}
//-----------------------------------------------------------------------------
TableSparsity::Iterator::~Iterator() 
{ 
  // Do nothing
}
//-----------------------------------------------------------------------------
TableSparsity::Iterator& TableSparsity::Iterator::operator++()
{
  if ( pos == (s.list[i].size() - 1) )
    at_end = true;
  else if ( s.list[i](pos+1) == -1 )
    at_end = true;
  else
    pos++;
  
  return *this;
}
//-----------------------------------------------------------------------------
int TableSparsity::Iterator::operator*() const
{
  return s.list[i](pos);
}
//-----------------------------------------------------------------------------
bool TableSparsity::Iterator::end() const
{
  return at_end;
}
//-----------------------------------------------------------------------------
