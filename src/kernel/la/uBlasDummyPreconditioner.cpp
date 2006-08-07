// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-07-04
// Last changed: 2006-07-04

#include <dolfin/uBlasVector.h>
#include <dolfin/uBlasDummyPreconditioner.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
uBlasDummyPreconditioner::uBlasDummyPreconditioner() : uBlasPreconditioner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBlasDummyPreconditioner::~uBlasDummyPreconditioner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void uBlasDummyPreconditioner::solve(uBlasVector& x, const uBlasVector& b) const
{
  x.assign(b);
}
//-----------------------------------------------------------------------------
