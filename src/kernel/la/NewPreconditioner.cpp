// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/NewPreconditioner.h>
#include <dolfin/NewVector.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewPreconditioner::NewPreconditioner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewPreconditioner::~NewPreconditioner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
int NewPreconditioner::PCApply(PC pc, Vec x, Vec y)
{
  NewPreconditioner* newpc = (NewPreconditioner*)pc->data;

  NewVector dolfinx(x), dolfiny(y);

  newpc->solve(dolfinx, dolfiny);

  return 0;
}
