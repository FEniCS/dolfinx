// Copyright (C) 2008-2017 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <petscsys.h>
#include <string>

namespace dolfinx
{

namespace common
{

/// This is a singleton class which manages the initialisation and
/// finalisation of various sub systems, such as MPI and PETSc.

class SubSystemsManager
{
public:
  /// Singleton instance. Calling this ensures singleton instance of
  /// SubSystemsManager is initialized according to the "Construct on
  /// First Use" idiom.
  static SubSystemsManager& singleton();

  // Copy constructor
  SubSystemsManager(const SubSystemsManager&) = delete;

  /// Initialise MPI
  static void init_mpi();

  /// Initialise MPI with required level of thread support
  static int init_mpi(int argc, char* argv[], int required_thread_level);

  /// Initialise loguru
  static void init_logging(int argc, char* argv[]);

  /// Initialize PETSc without command-line arguments
  static void init_petsc();

  /// Initialize PETSc with command-line arguments. Note that PETSc
  /// command-line arguments may also be filtered and sent to PETSc by
  /// parameters.parse(argc, argv).
  static void init_petsc(int argc, char* argv[]);

  /// Finalize subsystems. This will be called by the destructor, but in
  /// special cases it may be necessary to call finalize() explicitly.
  static void finalize();

  /// Return true if DOLFINX initialised MPI (and is therefore
  /// responsible for finalization)
  static bool responsible_mpi();

  /// Return true if DOLFINX initialised PETSc (and is therefore
  /// responsible for finalization)
  static bool responsible_petsc();

  /// Check if MPI has been initialised (returns true if MPI has been
  /// initialised, even if it is later finalised)
  static bool mpi_initialized();

  /// Check if MPI has been finalized (returns true if MPI has been
  /// finalised)
  static bool mpi_finalized();

  /// PETSc error handler. Logs everything known to DOLFINX logging
  /// system (with level TRACE) and stores the error message into
  /// pests_err_msg member.
  static PetscErrorCode
  PetscDolfinErrorHandler(MPI_Comm comm, int line, const char* fun,
                          const char* file, PetscErrorCode n, PetscErrorType p,
                          const char* mess, void* ctx);

  /// Last recorded PETSc error message
  std::string petsc_err_msg;

private:
  // Constructor (private)
  SubSystemsManager();

  // Destructor
  ~SubSystemsManager();

  // Finalize MPI
  static void finalize_mpi();

  // Finalize PETSc
  static void finalize_petsc();

  // State variables
  bool petsc_initialized;
  bool control_mpi;
};
} // namespace common
} // namespace dolfinx
