// Copyright (C) 2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-01-07
// Last changed: 

#ifdef HAVE_PETSC_H
#include <petsc.h>
#endif

#ifdef HAVE_SLEPC_H
#include <slepc.h>
#endif

#ifdef HAVE_MPI_H
#include <mpi.h>
#endif

#include <dolfin/dolfin_log.h>
#include <dolfin/SubSystemsManager.h>

using namespace dolfin;

// Initialise static data
dolfin::SubSystemsManager dolfin::SubSystemsManager::sub_systems_manager;

//-----------------------------------------------------------------------------
SubSystemsManager::SubSystemsManager() : petsc_initialized(false),
                                         petsc_controls_mpi(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
SubSystemsManager::SubSystemsManager(const SubSystemsManager& sub_sys_manager)
{
  error("Should not be using copy constructor of SubSystemsManager.");
}
//-----------------------------------------------------------------------------
SubSystemsManager::~SubSystemsManager()
{
  // Finalize subsystems in the correct order
  finalizePETSc();
  finalizeMPI();
}
//-----------------------------------------------------------------------------
void SubSystemsManager::initMPI()
{
  // Initialize MPI
  MPIinit();
}
//-----------------------------------------------------------------------------
#ifdef HAVE_PETSC_H
void SubSystemsManager::initPETSc()
{
  if ( sub_systems_manager.petsc_initialized )
    return;

  message("Initializing PETSc (ignoring command-line arguments).");

  // Prepare fake command-line arguments for PETSc. This is needed since
  // PetscInitializeNoArguments() does not seem to work.
  int argc = 1;
  char** argv = new char * [1];
  argv[0] = new char[DOLFIN_LINELENGTH];
  snprintf(argv[0], DOLFIN_LINELENGTH, "%s", "unknown");

  // Initialize PETSc
  initPETSc(argc, argv, false);

  // Cleanup
  delete [] argv[0];
  delete [] argv;
}
//-----------------------------------------------------------------------------
void SubSystemsManager::initPETSc(int argc, char* argv[], bool cmd_line_args)
{
  if ( sub_systems_manager.petsc_initialized )
    return;

  // Get status of MPI before PETSc initialisation
  const bool mpi_init_status = MPIinitialized();

  if(cmd_line_args)
    message("Initializing PETSc with given command-line arguments.");

  // Initialize PETSc
  PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);

#ifdef HAVE_SLEPC_H
  // Initialize SLEPc
  SlepcInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
#endif

  sub_systems_manager.petsc_initialized = true;

  // Determine if PETSc initialised MPI and is then responsible for MPI finalization
  if(!mpi_init_status && MPIinitialized())
    sub_systems_manager.petsc_controls_mpi = true;
}
//-----------------------------------------------------------------------------
#else
void SubSystemsManager::initPETSc()
{
  error("DOLFIN has not been configured for PETSc.");
}
//-----------------------------------------------------------------------------
void SubSystemsManager::initPETSc(int argc, char* argv[], bool cmd_line_args)
{
  error("DOLFIN has not been configured for PETSc.");
}
#endif
//-----------------------------------------------------------------------------
void SubSystemsManager::finalizeMPI()
{
#ifdef HAVE_MPI_H
  // Finalise MPI if required
  if ( MPIinitialized() )
  {
    dolfin_debug("Finalizing MPI");
    MPI_Finalize();
  }
#else
  // Do nothing
#endif
}
//-----------------------------------------------------------------------------
void SubSystemsManager::finalizePETSc()
{
#ifdef HAVE_PETSC_H
 if ( sub_systems_manager.petsc_initialized )
  {
    dolfin_debug("Finalizing PETSc");
    PetscFinalize();
 
    #ifdef HAVE_SLEPC_H
    SlepcFinalize();
    #endif
  }
#else
  // Do nothing
#endif
}
//-----------------------------------------------------------------------------
bool SubSystemsManager::MPIinitialized()
{
#ifdef HAVE_MPI_H
  int initialized;
  MPI_Initialized(&initialized);
  if (initialized)
    return true;
  else if (!initialized)
    return false;
  else
  {
    error("MPI_Initialized has returned an unknown initialization status");
    return false;
  }
#else
  // DOLFIN is not configured for MPI (it might be through PETSc)
  return false;
#endif
}
//-----------------------------------------------------------------------------
void SubSystemsManager::MPIinit()
{
#ifdef HAVE_MPI_H
  if( MPIinitialized() )
    return;

  dolfin_debug("Initializing MPI");
  MPI_Init(0, 0);
#else
  // Do nothing
#endif
}
//-----------------------------------------------------------------------------



