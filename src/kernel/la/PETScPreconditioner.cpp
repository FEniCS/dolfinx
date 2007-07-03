// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2005.
// Modified by Anders Logg 2006.
//
// First added:  2005
// Last changed: 2006-08-15

#ifdef HAVE_PETSC_H

#include <dolfin/PETScManager.h>
#if PETSC_VERSION_MAJOR==2 && PETSC_VERSION_MINOR==3 && PETSC_VERSION_SUBMINOR==0
  #include <src/ksp/pc/pcimpl.h>
#else
  #include <private/pcimpl.h>
#endif

#include <dolfin/PETScPreconditioner.h>
#include <dolfin/Vector.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
PETScPreconditioner::PETScPreconditioner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScPreconditioner::~PETScPreconditioner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void PETScPreconditioner::setup(const KSP ksp, PETScPreconditioner &pc)
{
  PC petscpc;
  KSPGetPC(ksp, &petscpc);

  PETScPreconditioner::PCCreate(petscpc);

  petscpc->data = &pc;
  petscpc->ops->apply = PETScPreconditioner::PCApply;
  petscpc->ops->applytranspose = PETScPreconditioner::PCApply;
  petscpc->ops->applysymmetricleft = PETScPreconditioner::PCApply;
  petscpc->ops->applysymmetricright = PETScPreconditioner::PCApply;
}
//-----------------------------------------------------------------------------
int PETScPreconditioner::PCApply(PC pc, Vec x, Vec y)
{
  // Convert vectors to DOLFIN wrapper format and pass to DOLFIN preconditioner

  PETScPreconditioner* newpc = (PETScPreconditioner*)pc->data;

  PETScVector dolfinx(x), dolfiny(y);

  newpc->solve(dolfiny, dolfinx);

  return 0;
}
//-----------------------------------------------------------------------------
int PETScPreconditioner::PCCreate(PC pc)
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
PCType PETScPreconditioner::getType(Preconditioner pc)
{
  switch (pc)
  {
  case default_pc:
    return "default";
  case amg:
    return PCHYPRE;
  case icc:
    return PCICC;
  case ilu:
    return PCILU;
  case jacobi:
    return PCJACOBI;
  case sor:
    return PCSOR;
  case none:
    return PCNONE;
  default:
    warning("Requested preconditioner unkown. Using incomplete LU.");
    return PCILU;
  }
}
//-----------------------------------------------------------------------------

#endif
