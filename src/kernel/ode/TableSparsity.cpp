// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/TableSparsity.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
TableSparsity::TableSparsity(unsigned int N) : GenericSparsity(N)
{
  list.resize(N);
}
//-----------------------------------------------------------------------------
TableSparsity::~TableSparsity()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void TableSparsity::setsize(unsigned int i, unsigned int size)
{
  dolfin_assert(i >= 0);
  dolfin_assert(i < N);

  if ( size > N )
    dolfin_error("Number of dependencies cannot be larger than system size.");

  list[i].resize(size);

  // Index -1 denotes an empty position
  for (unsigned int pos = 0; pos < size; pos++)
    list[i][pos] = -1;
}
//-----------------------------------------------------------------------------
void TableSparsity::set(unsigned int i, unsigned int j)
{
  dolfin_assert(i >= 0);
  dolfin_assert(i < N);

  int col = static_cast<int>(j);

  // Find first empty position
  for (unsigned int pos = 0; pos < list[i].size(); pos++) {
    if ( list[i][pos] == col )
      return;
    else if ( list[i][pos] == -1 ) {
      list[i][pos] = col;
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
TableSparsity::Iterator* TableSparsity::createIterator(unsigned int i) const
{
  dolfin_assert(i >= 0);
  dolfin_assert(i < N);

  return new Iterator(i, *this);
}
//-----------------------------------------------------------------------------
TableSparsity::Iterator::Iterator(unsigned int i, const TableSparsity& sparsity)
  : GenericSparsity::Iterator(i), s(sparsity)
{
  pos = 0;

  if ( s.list[i].size() == 0 )
    at_end = true;
  else if ( s.list[i][0] == -1 )
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
  else if ( s.list[i][pos+1] == -1 )
    at_end = true;
  else
    pos++;
  
  return *this;
}
//-----------------------------------------------------------------------------
unsigned int TableSparsity::Iterator::operator*() const
{
  return static_cast<unsigned int>(s.list[i][pos]);
}
//-----------------------------------------------------------------------------
bool TableSparsity::Iterator::end() const
{
  return at_end;
}
//-----------------------------------------------------------------------------
