// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <iostream>
#include <petsc/petsc.h>
#include <dolfin/PETScManager.h>

// Initialize static data
dolfin::PETScManager dolfin::PETScManager::petsc;

using namespace dolfin;

//-----------------------------------------------------------------------------
void PETScManager::init()
{
  // Do nothing, just need to make sure this piece of code is called
  // so that the static data is initialized
}
//-----------------------------------------------------------------------------
PETScManager::PETScManager()
{
  std::cout << "Initializing PETSc" << std::endl;
  CHKERRQ(PetscInitializeNoArguments());
}
//-----------------------------------------------------------------------------
PETScManager::~PETScManager()
{
  std::cout << "Finalizing PETSc" << std::endl;
  CHKERRQ(PetscFinalize());
}
//-----------------------------------------------------------------------------
