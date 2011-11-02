// Copyright (C) 2008 Garth N. Wells
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
// Modified by Anders Logg, 2008.
//
// First added:  2008-01-07
// Last changed: 2011-03-17

#ifdef HAS_MPI
#include <mpi.h>
#include <iostream>
#endif

#ifdef HAS_PETSC
#include <petsc.h>
#endif

#ifdef HAS_SLEPC
#include <slepc.h>
#endif

#include <libxml/parser.h>
#include <dolfin/common/constants.h>
#include <dolfin/log/dolfin_log.h>
#include "SubSystemsManager.h"

using namespace dolfin;


// Return singleton instance. Do NOT make the singleton a global static object;
// the method here ensures that the singleton is initialised before use.
// (google "static initialization order fiasco" for full explanation)

SubSystemsManager& SubSystemsManager::singleton()
{
  static SubSystemsManager the_instance;
  return the_instance;
}
//-----------------------------------------------------------------------------
SubSystemsManager::SubSystemsManager() : petsc_initialized(false),
                                         control_mpi(false)
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
  finalize();
}
//-----------------------------------------------------------------------------
void SubSystemsManager::init_mpi()
{
  #ifdef HAS_MPI
  if( MPI::Is_initialized() )
    return;

  // Initialise MPI and take responsibility
  MPI::Init();
  singleton().control_mpi = true;
  #else
  // Do nothing
  #endif
}
//-----------------------------------------------------------------------------
void SubSystemsManager::init_mpi_threaded(int argc, char* argv[])
{
  #ifdef HAS_MPI
  std::cout << "Inside thread init" << std::endl;
  if( MPI::Is_initialized() )
    return;

  std::cout << "Init thread (1)" << std::endl;


  // Initialise MPI and take responsibility
  int required = MPI_THREAD_MULTIPLE;
  int provided = -1;
  MPI_Init_thread(&argc, &argv, required, &provided);
  singleton().control_mpi = true;

  switch (provided)
    {
    case MPI_THREAD_SINGLE:
      printf("MPI_Init_thread level = MPI_THREAD_SINGLE\n");
      break;
    case MPI_THREAD_FUNNELED:
      printf("MPI_Init_thread level = MPI_THREAD_FUNNELED\n");
      break;
    case MPI_THREAD_SERIALIZED:
      printf("MPI_Init_thread level = MPI_THREAD_SERIALIZED\n");
      break;
    case MPI_THREAD_MULTIPLE:
      printf("MPI_Init_thread level = MPI_THREAD_MULTIPLE\n");
      break;
    default:
      printf("MPI_Init_thread level = ???\n");
    }
  #else
  // Do nothing
  #endif
}
//-----------------------------------------------------------------------------
void SubSystemsManager::init_petsc()
{
#ifdef HAS_PETSC
  if ( singleton().petsc_initialized )
    return;

  log(TRACE, "Initializing PETSc (ignoring command-line arguments).");

  // Dummy command-line arguments for PETSc. This is needed since
  // PetscInitializeNoArguments() does not seem to work.
  int argc = 0;
  char** argv = NULL;

  // Initialize PETSc
  init_petsc(argc, argv);
#else
  error("DOLFIN has not been configured for PETSc.");
#endif
}
//-----------------------------------------------------------------------------
void SubSystemsManager::init_petsc(int argc, char* argv[])
{
#ifdef HAS_PETSC
  if ( singleton().petsc_initialized )
    return;

  // Get status of MPI before PETSc initialisation
  const bool mpi_init_status = mpi_initialized();

  // Print message if PETSc is intialised with command line arguments
  if (argc > 1)
    log(TRACE, "Initializing PETSc with given command-line arguments.");

  // Initialize PETSc
  PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);

#ifdef HAS_SLEPC
  // Initialize SLEPc
  SlepcInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
#endif

  singleton().petsc_initialized = true;

  // Determine if PETSc initialised MPI (and is therefore responsible for MPI finalization)
  if (mpi_initialized() and !mpi_init_status)
    singleton().control_mpi = false;
#else
  error("DOLFIN has not been configured for PETSc.");
#endif
}
//-----------------------------------------------------------------------------
void SubSystemsManager::finalize()
{
  // Finalize subsystems in the correct order
  finalize_petsc();
  finalize_mpi();

  // Clean up libxml2 parser
  xmlCleanupParser();
}
//-----------------------------------------------------------------------------
bool SubSystemsManager::responsible_mpi()
{
  return singleton().control_mpi;
}
//-----------------------------------------------------------------------------
bool SubSystemsManager::responsible_petsc()
{
  return singleton().petsc_initialized;
}
//-----------------------------------------------------------------------------
void SubSystemsManager::finalize_mpi()
{
#ifdef HAS_MPI
  // Finalise MPI if required
  if (MPI::Is_initialized() and singleton().control_mpi)
  {
    // Check in MPI has already been finalised (possibly incorrectly by a
    // 3rd party libary). Is it hasn't, finalise as normal.
    if (!MPI::Is_finalized())
      MPI::Finalize();
    else
    {
      // Use std::cout since log system may fail because MPI has been shut down.
      std::cout << "DOLFIN is responsible for MPI, but it has been finalized elsewhere prematurely." << std::endl;
      std::cout << "This is usually due to a bug in a 3rd party library, and can lead to unpredictable behaviour." << std::endl;
      std::cout << "If using PyTrilinos, make sure that PyTrilinos modules are imported before the DOLFIN module." << std::endl;
    }

    singleton().control_mpi = false;
  }
#else
  // Do nothing
#endif
}
//-----------------------------------------------------------------------------
void SubSystemsManager::finalize_petsc()
{
#ifdef HAS_PETSC
  if (singleton().petsc_initialized)
  {
    PetscFinalize();
    singleton().petsc_initialized = false;

    #ifdef HAS_SLEPC
    SlepcFinalize();
    #endif
  }
#else
  // Do nothing
#endif
}
//-----------------------------------------------------------------------------
bool SubSystemsManager::mpi_initialized()
{
  // This function not affected if MPI_Finalize has been called. It returns
  // true if MPI_Init has been called at any point, even if MPI_Finalize has
  // been called.

#ifdef HAS_MPI
  return MPI::Is_initialized();
#else
  // DOLFIN is not configured for MPI (it might be through PETSc)
  return false;
#endif
}
//-----------------------------------------------------------------------------
bool SubSystemsManager::mpi_finalized()
{
#ifdef HAS_MPI
  return MPI::Is_finalized();
#else
  // DOLFIN is not configured for MPI (it might be through PETSc)
  return false;
#endif
}
//-----------------------------------------------------------------------------
