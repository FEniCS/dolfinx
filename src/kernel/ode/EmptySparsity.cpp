// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/EmptySparsity.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
EmptySparsity::EmptySparsity(unsigned int N) : GenericSparsity(N)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
EmptySparsity::~EmptySparsity()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
GenericSparsity::Type EmptySparsity::type() const
{
  return empty;
}
//-----------------------------------------------------------------------------
EmptySparsity::Iterator* EmptySparsity::createIterator(unsigned int i) const
{
  dolfin_assert(i >= 0);
  dolfin_assert(i < N);

  return new Iterator(i, *this);
}
//-----------------------------------------------------------------------------
EmptySparsity::Iterator::Iterator(unsigned int i, const EmptySparsity& sparsity) 
  : GenericSparsity::Iterator(i), s(sparsity)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
EmptySparsity::Iterator::~Iterator()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
EmptySparsity::Iterator& EmptySparsity::Iterator::operator++()
{
  return *this;
}
//-----------------------------------------------------------------------------
unsigned int EmptySparsity::Iterator::operator*() const
{
  return 0;
}
//-----------------------------------------------------------------------------
bool EmptySparsity::Iterator::end() const
{
  return true;
}
//-----------------------------------------------------------------------------
