// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/IndexPair.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
IndexPair::IndexPair() : i(0), j(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
IndexPair::IndexPair(unsigned int i, unsigned int j) : i(i), j(j)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
IndexPair::~IndexPair()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
