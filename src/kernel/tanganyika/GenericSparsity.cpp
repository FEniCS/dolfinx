// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/GenericSparsity.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericSparsity::GenericSparsity(int N)
{
  this->N = N;
}
//-----------------------------------------------------------------------------
GenericSparsity::~GenericSparsity()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void GenericSparsity::setsize(int i, int size)
{
  dolfin_error("Explicit dependencies can only be specified for table sparsity.");
}
//-----------------------------------------------------------------------------
void GenericSparsity::set(int i, int j)
{
  dolfin_error("Explicit dependencies can only be specified for table sparsity.");
}
//-----------------------------------------------------------------------------
GenericSparsity::Iterator::Iterator(int i)
{
  this->i = i;
}
//-----------------------------------------------------------------------------
GenericSparsity::Iterator::~Iterator()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
