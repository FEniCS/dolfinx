// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/GenericSparsity.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericSparsity::GenericSparsity(unsigned int N)
{
  this->N = N;
}
//-----------------------------------------------------------------------------
GenericSparsity::~GenericSparsity()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void GenericSparsity::setsize(unsigned int i, unsigned int size)
{
  dolfin_error("Explicit dependencies can only be specified for table sparsity.");
}
//-----------------------------------------------------------------------------
void GenericSparsity::set(unsigned int i, unsigned int j)
{
  dolfin_error("Explicit dependencies can only be specified for table sparsity.");
}
//-----------------------------------------------------------------------------
GenericSparsity::Iterator::Iterator(unsigned int i)
{
  this->i = i;
}
//-----------------------------------------------------------------------------
GenericSparsity::Iterator::~Iterator()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
