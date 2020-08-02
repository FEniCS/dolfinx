// Copyright (C) 2008-2020 Garth N. Wells, Anders Logg, Jan Blechta
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "SubSystemsManager.h"
#include <dolfinx/common/log.h>
#include <iostream>
#include <mpi.h>
#include <petscsys.h>
#include <string>

#ifdef HAS_SLEPC
#include <slepcsys.h>
#endif

using namespace dolfinx::common;

//-----------------------------------------------------------------------------
void SubSystemsManager::init_mpi()
{
  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized)
    return;

  // Init MPI
  std::string s("");
  char* c = const_cast<char*>(s.c_str());
  SubSystemsManager::init_mpi(0, &c);
}
//-----------------------------------------------------------------------------
void SubSystemsManager::init_mpi(int argc, char* argv[])
{
  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized)
    return;

  // Initialise MPI
  MPI_Init(&argc, &argv);
}
//-----------------------------------------------------------------------------
void SubSystemsManager::init_logging(int argc, char* argv[])
{
  loguru::g_stderr_verbosity = loguru::Verbosity_WARNING;
  loguru::Options options = {"-dolfinx_loglevel", "main thread", false};
  loguru::init(argc, argv, options);
}
//-----------------------------------------------------------------------------
void SubSystemsManager::init_petsc()
{
  int argc = 0;
  char** argv = nullptr;
  init_petsc(argc, argv);
}
//-----------------------------------------------------------------------------
void SubSystemsManager::init_petsc(int argc, char* argv[])
{
  if (argc > 1)
    LOG(INFO) << "Initializing PETSc with given command-line arguments.";

  PetscBool is_initialized;
  PetscInitialized(&is_initialized);
  if (!is_initialized)
    PetscInitialize(&argc, &argv, nullptr, nullptr);

#ifdef HAS_SLEPC
  SlepcInitialize(&argc, &argv, nullptr, nullptr);
#endif
}
//-----------------------------------------------------------------------------
void SubSystemsManager::finalize_mpi()
{
  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized)
  {
    // Check if MPI has already been finalised (possibly incorrectly by
    // a 3rd party library). If it hasn't, finalise as normal.
    int mpi_finalized;
    MPI_Finalized(&mpi_finalized);
    if (!mpi_finalized)
      MPI_Finalize();
    else
    {
      // Use std::cout since log system may fail because MPI has been shut down.
      std::cout << "MPI has already been finalised, possibly prematurely by "
                   "another library."
                << std::endl;
    }
  }
}
//-----------------------------------------------------------------------------
void SubSystemsManager::finalize_petsc()
{
  PetscFinalize();
#ifdef HAS_SLEPC
  SlepcFinalize();
#endif
}
//-----------------------------------------------------------------------------
bool SubSystemsManager::mpi_initialized()
{
  // This function is not affected if MPI_Finalize has been called. It
  // returns true if MPI_Init has been called at any point, even if
  // MPI_Finalize has been called.
  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);
  return mpi_initialized;
}
//-----------------------------------------------------------------------------
bool SubSystemsManager::mpi_finalized()
{
  int mpi_finalized;
  MPI_Finalized(&mpi_finalized);
  return mpi_finalized;
}
//-----------------------------------------------------------------------------
