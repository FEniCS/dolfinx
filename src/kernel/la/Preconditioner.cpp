// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005
// Last changed: 2005

#include <src/ksp/pc/pcimpl.h>
#include <dolfin/Preconditioner.h>
#include <dolfin/Vector.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Preconditioner::Preconditioner()
{
  // Do nothing

}
//-----------------------------------------------------------------------------
Preconditioner::~Preconditioner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
int Preconditioner::PCApply(PC pc, Vec x, Vec y)
{
  // Convert vectors to DOLFIN wrapper format and pass to DOLFIN preconditioner

  Preconditioner* newpc = (Preconditioner*)pc->data;

  Vector dolfinx(x), dolfiny(y);

  newpc->solve(dolfiny, dolfinx);

  return 0;
}
//-----------------------------------------------------------------------------
int Preconditioner::PCCreate(PC pc)
{
  // Initialize function pointers to 0

  pc->ops->setup               = 0;
  pc->ops->apply               = 0;
  pc->ops->applyrichardson     = 0;
  pc->ops->applytranspose      = 0;
  pc->ops->applysymmetricleft  = 0;
  pc->ops->applysymmetricright = 0;
  pc->ops->setfromoptions      = 0;
  pc->ops->view                = 0;
  pc->ops->destroy             = 0;

  // Set PETSc name of preconditioner
  PetscObjectChangeTypeName((PetscObject)pc, "DOLFIN");

  return 0;
}
//-----------------------------------------------------------------------------
