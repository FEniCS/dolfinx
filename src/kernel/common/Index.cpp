// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Index.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Index::Index() : i(0), j(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Index::Index(unsigned int i, unsigned int j) : i(i), j(j)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Index::~Index()
{
  // Do nothing
}
//-----------------------------------------------------------------------------

