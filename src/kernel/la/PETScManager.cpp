// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <stdio.h>
#include <iostream>
#include <petsc/petsc.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/PETScManager.h>

// Initialize static data
dolfin::PETScManager dolfin::PETScManager::petsc;

using namespace dolfin;

//-----------------------------------------------------------------------------
void PETScManager::init()
{
  if ( petsc.initialized )
    return;

  cout << "Initializing PETSc (ignoring command-line arguments)." << endl;

  // Prepare fake command-line arguments for PETSc. This is needed since
  // PetscInitializeNoArguments() does not seem to work.
  int argc = 1;
  char** argv = new char*[1];
  argv[0] = new char[7];
  sprintf(argv[0], "%s", "uknown");

  // Initialize PETSc
  PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);

  // Cleanup
  delete [] argv[0];
  delete [] argv;

  petsc.initialized = true;
}
//-----------------------------------------------------------------------------
void PETScManager::init(int argc, char* argv[])
{
  if ( petsc.initialized )
    return;

  cout << "Initializing PETSc (with given command-line arguments)." << endl;

  // Initialize PETSc
  PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);

  petsc.initialized = true;
}
//-----------------------------------------------------------------------------
PETScManager::PETScManager() : initialized(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PETScManager::~PETScManager()
{
  if ( petsc.initialized )
  {
    std::cout << "Finalizing PETSc." << std::endl;
    PetscFinalize();
  }
}
//-----------------------------------------------------------------------------
