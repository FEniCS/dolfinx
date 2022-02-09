// Copyright (C) 2008-2020 Garth N. Wells, Anders Logg, Jan Blechta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "subsystem.h"
#include <dolfinx/common/log.h>
#include <iostream>
#include <mpi.h>
#include <petscsys.h>
#include <string>
#include <vector>

#ifdef HAS_SLEPC
#include <slepcsys.h>
#endif

using namespace dolfinx::common;

//-----------------------------------------------------------------------------
void subsystem::init_mpi()
{
  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized)
    return;

  // Init MPI
  std::string s("");
  char* c = const_cast<char*>(s.c_str());
  subsystem::init_mpi(0, &c);
}
//-----------------------------------------------------------------------------
void subsystem::init_mpi(int argc, char* argv[])
{
  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized)
    return;

  // Initialise MPI
  MPI_Init(&argc, &argv);
}
//-----------------------------------------------------------------------------
void subsystem::init_logging(int argc, char* argv[])
{
  loguru::g_stderr_verbosity = loguru::Verbosity_WARNING;

#ifndef NDEBUG
  loguru::SignalOptions signals;
#else
  loguru::SignalOptions signals = loguru::SignalOptions::none();
#endif

  loguru::Options options = {"-dolfinx_loglevel", "main", signals};

  // Make a copy of argv, as loguru may modify it.
  std::vector<char*> argv_copy;
  for (int i = 0; i < argc; ++i)
    argv_copy.push_back(argv[i]);
  argv_copy.push_back(nullptr);

  loguru::init(argc, argv_copy.data(), options);
}
//-----------------------------------------------------------------------------
void subsystem::init_petsc()
{
  int argc = 0;
  char** argv = nullptr;
  init_petsc(argc, argv);
}
//-----------------------------------------------------------------------------
void subsystem::init_petsc(int argc, char* argv[])
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
void subsystem::finalize_mpi()
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
void subsystem::finalize_petsc()
{
  PetscFinalize();
#ifdef HAS_SLEPC
  SlepcFinalize();
#endif
}
//-----------------------------------------------------------------------------
bool subsystem::mpi_initialized()
{
  // This function is not affected if MPI_Finalize has been called. It
  // returns true if MPI_Init has been called at any point, even if
  // MPI_Finalize has been called.
  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);
  return mpi_initialized;
}
//-----------------------------------------------------------------------------
bool subsystem::mpi_finalized()
{
  int mpi_finalized;
  MPI_Finalized(&mpi_finalized);
  return mpi_finalized;
}
//-----------------------------------------------------------------------------
