// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/FullSparsity.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
FullSparsity::FullSparsity(unsigned int N) : GenericSparsity(N)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FullSparsity::~FullSparsity()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
GenericSparsity::Type FullSparsity::type() const
{
  return full;
}
//-----------------------------------------------------------------------------
FullSparsity::Iterator* FullSparsity::createIterator(unsigned int i) const
{
  dolfin_assert(i >= 0);
  dolfin_assert(i < N);

  return new Iterator(i, *this);
}
//-----------------------------------------------------------------------------
FullSparsity::Iterator::Iterator(unsigned int i, const FullSparsity& sparsity) 
  : GenericSparsity::Iterator(i), s(sparsity)
{
  pos = 0;
  at_end = s.N <= 1;
}
//-----------------------------------------------------------------------------
FullSparsity::Iterator::~Iterator()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FullSparsity::Iterator& FullSparsity::Iterator::operator++()
{
  if ( pos == (s.N - 1) )
    at_end = true;
  else
    pos++;
  
  return *this;
}
//-----------------------------------------------------------------------------
unsigned int FullSparsity::Iterator::operator*() const
{
  return pos;
}
//-----------------------------------------------------------------------------
bool FullSparsity::Iterator::end() const
{
  return at_end;
}
//-----------------------------------------------------------------------------
