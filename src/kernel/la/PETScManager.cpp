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

  // Prepare fake command-line arguments for PETSc. This is needed since
  // PetscInitializeNoArguments() does not seem to work.
  int argc = 0;
  char** argv = new char*[1];
  argv[0] = new char[1];
  argv[0][0] = '\0';

  // Initialize PETSc
  PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);

  // Cleanup
  delete [] argv[0];
  delete [] argv;
}
//-----------------------------------------------------------------------------
PETScManager::~PETScManager()
{
  std::cout << "Finalizing PETSc" << std::endl;
  PetscFinalize();
}
//-----------------------------------------------------------------------------
