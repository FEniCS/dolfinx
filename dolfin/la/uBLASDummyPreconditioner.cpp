// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-07-04
// Last changed: 2006-07-04

#include "uBLASVector.h"
#include "uBLASDummyPreconditioner.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
uBLASDummyPreconditioner::uBLASDummyPreconditioner() : uBLASPreconditioner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBLASDummyPreconditioner::~uBLASDummyPreconditioner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void uBLASDummyPreconditioner::solve(uBLASVector& x, const uBLASVector& b) const
{
  x.vec().assign(b.vec());
}
//-----------------------------------------------------------------------------
