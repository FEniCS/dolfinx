// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/PropagatingPreconditioner.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
PropagatingPreconditioner::PropagatingPreconditioner(const NewJacobianMatrix& A, 
						     NewTimeSlab& timeslab)
  : A(A), ts(timeslab)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PropagatingPreconditioner::~PropagatingPreconditioner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void PropagatingPreconditioner::solve(NewVector& x, const NewVector& b) const
{
  x = b;
}
//-----------------------------------------------------------------------------
