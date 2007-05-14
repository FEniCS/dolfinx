// Copyright (C) 2004-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2004-12-09
// Last changed: 2006-05-15

#ifdef HAVE_PETSC_H

#include <dolfin/PETScManager.h>

#include <stdio.h>

#ifdef HAVE_SLEPC_H
#include <slepc.h>
#endif

#include <dolfin/dolfin_log.h>
#include <dolfin/constants.h>

// Initialize static data
dolfin::PETScManager dolfin::PETScManager::petsc;

using namespace dolfin;

//-----------------------------------------------------------------------------
void PETScManager::init()
{
  if ( petsc.initialized )
    return;

  message("Initializing PETSc (ignoring command-line arguments).");

  // Prepare fake command-line arguments for PETSc. This is needed since
  // PetscInitializeNoArguments() does not seem to work.
  int argc = 1;
  char** argv = new char * [1];
  argv[0] = new char[DOLFIN_WORDLENGTH];
  sprintf(argv[0], "%s", "unknown");

  // Initialize PETSc
  PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);

#ifdef HAVE_SLEPC_H
  // Initialize SLEPc
  SlepcInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
#endif


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

  message("Initializing PETSc with given command-line arguments.");
  
  // Initialize PETSc
  PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);

  #ifdef HAVE_SLEPC_H
  // Initialize SLEPc
  SlepcInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
  #endif

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
    // FIXME: Can't use log system here because it may already
    // FIXME: be out of scope/destroyed
    //message("Finalizing PETSc.");
    //printf("Finalizing PETSc.\n");
    PetscFinalize();
 
    #ifdef HAVE_SLEPC_H
    SlepcFinalize();
    #endif

  }
}
//-----------------------------------------------------------------------------

#endif
