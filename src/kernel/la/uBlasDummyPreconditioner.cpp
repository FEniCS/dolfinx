// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-07-04
// Last changed: 2006-07-04

#include <dolfin/DenseVector.h>
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
void uBlasDummyPreconditioner::solve(DenseVector& x, const DenseVector& b) const
{
  x.assign(b);
}
//-----------------------------------------------------------------------------
