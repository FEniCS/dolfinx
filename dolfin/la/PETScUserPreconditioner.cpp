// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2005.
// Modified by Anders Logg 2006.
//
// First added:  2005
// Last changed: 2006-08-15

#ifdef HAS_PETSC

#include <boost/shared_ptr.hpp>
#include <private/pcimpl.h>
#include <dolfin/common/NoDeleter.h>
#include "PETScVector.h"
#include "PETScUserPreconditioner.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
PETScUserPreconditioner::PETScUserPreconditioner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScUserPreconditioner::~PETScUserPreconditioner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void PETScUserPreconditioner::setup(const KSP ksp, PETScUserPreconditioner& pc)
{
  PC petscpc;
  KSPGetPC(ksp, &petscpc);

  PETScUserPreconditioner::PCCreate(petscpc);

  petscpc->data = &pc;
  petscpc->ops->apply = PETScUserPreconditioner::PCApply;
  petscpc->ops->applytranspose = PETScUserPreconditioner::PCApply;
  petscpc->ops->applysymmetricleft = PETScUserPreconditioner::PCApply;
  petscpc->ops->applysymmetricright = PETScUserPreconditioner::PCApply;
}
//-----------------------------------------------------------------------------
int PETScUserPreconditioner::PCApply(PC pc, Vec x, Vec y)
{
  // Convert vectors to DOLFIN wrapper format and pass to DOLFIN preconditioner

  PETScUserPreconditioner* newpc = (PETScUserPreconditioner*)pc->data;

  boost::shared_ptr<Vec> _x(&x, NoDeleter<Vec>());
  boost::shared_ptr<Vec> _y(&y, NoDeleter<Vec>());
  PETScVector dolfinx(_x), dolfiny(_y);

  newpc->solve(dolfiny, dolfinx);

  return 0;
}
//-----------------------------------------------------------------------------
int PETScUserPreconditioner::PCCreate(PC pc)
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

#endif
