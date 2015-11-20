// Copyright (C) 2005 Johan Jansson
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells 2005, 2012.
// Modified by Anders Logg 2006.
//
// First added:  2005
// Last changed: 2012-05-10

#ifdef HAS_PETSC

#include <memory>

#include <petscversion.h>
#include <petsc/private/pcimpl.h>

#include <dolfin/common/NoDeleter.h>
#include "PETScVector.h"
#include "PETScUserPreconditioner.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
PETScUserPreconditioner::PETScUserPreconditioner():
  petscpc(NULL)
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
  PetscErrorCode ierr = KSPGetPC(ksp, &petscpc);
  if (ierr != 0) petsc_error(ierr, __FILE__, "KSPGetPC");
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
  // Convert vectors to DOLFIN wrapper format and pass to DOLFIN
  // preconditioner
  PETScUserPreconditioner* newpc = (PETScUserPreconditioner*)pc->data;

  // Wrap PETSc vectors as DOLFIN PETScVectors
  PETScVector dolfinx(x), dolfiny(y);

  // Solve
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
  pc->ops->reset               = 0;
  // Set PETSc name of preconditioner
  PetscObjectChangeTypeName((PetscObject)pc, "DOLFIN");

  return 0;
}
//-----------------------------------------------------------------------------

#endif
