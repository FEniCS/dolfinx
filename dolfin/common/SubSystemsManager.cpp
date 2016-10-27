// Copyright (C) 2008-2016 Garth N. Wells, Anders Logg, Jan Blechta
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

#ifdef HAS_MPI
#define MPICH_IGNORE_CXX_SEEK 1
#include <mpi.h>
#include <iostream>
#endif

#ifdef HAS_PETSC
#include <petsc.h>
#endif

#ifdef HAS_SLEPC
#include <slepc.h>
#endif

#include <boost/algorithm/string/trim.hpp>

#include <dolfin/common/constants.h>
#include <dolfin/common/Timer.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/log/log.h>
#include "SubSystemsManager.h"

using namespace dolfin;

// Return singleton instance. Do NOT make the singleton a global static
// object; the method here ensures that the singleton is initialised
// before use. (google "static initialization order fiasco" for full
// explanation)

SubSystemsManager& SubSystemsManager::singleton()
{
  static SubSystemsManager the_instance;
  return the_instance;
}
//-----------------------------------------------------------------------------
SubSystemsManager::SubSystemsManager() : petsc_err_msg(""),
  petsc_initialized(false), control_mpi(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
SubSystemsManager::SubSystemsManager(const SubSystemsManager& sub_sys_manager)
{
  dolfin_error("SubSystemsManager.cpp",
               "create subsystems manager",
               "Copy constructor should not be used");
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
  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized)
    return;

  // Init MPI with highest level of thread support and take responsibility
  std::string s("");
  char* c = const_cast<char *>(s.c_str());
  SubSystemsManager::init_mpi(0, &c, MPI_THREAD_MULTIPLE);
  singleton().control_mpi = true;
  #else
  // Do nothing
  #endif
}
//-----------------------------------------------------------------------------
int SubSystemsManager::init_mpi(int argc, char* argv[],
                                int required_thread_level)
{
  Timer timer("Init MPI");

  #ifdef HAS_MPI
  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized)
    return -100;

  // Initialise MPI and take responsibility
  int provided = -1;
  MPI_Init_thread(&argc, &argv, required_thread_level, &provided);
  singleton().control_mpi = true;

  const bool print_thread_support
    = dolfin::parameters["print_mpi_thread_support_level"];
  if (print_thread_support)
  {
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
        printf("MPI_Init_thread level = unknown\n");
      }
  }

  return provided;
  #else
  return -1;
  #endif
}
//-----------------------------------------------------------------------------
void SubSystemsManager::init_petsc()
{
  #ifdef HAS_PETSC
  if (singleton().petsc_initialized)
    return;

  log(TRACE, "Initializing PETSc (ignoring command-line arguments).");

  // Dummy command-line arguments for PETSc. This is needed since
  // PetscInitializeNoArguments() does not seem to work.
  int argc = 0;
  char** argv = NULL;

  // Initialize PETSc
  init_petsc(argc, argv);
  #else
  dolfin_error("SubSystemsManager.cpp",
               "initialize PETSc subsystem",
               "DOLFIN has not been configured with PETSc support");
  #endif
}
//-----------------------------------------------------------------------------
void SubSystemsManager::init_petsc(int argc, char* argv[])
{
  Timer timer("Init PETSc");

  #ifdef HAS_PETSC
  if (singleton().petsc_initialized)
    return;

  // Initialized MPI (do it here rather than letting PETSc do it to
  // make sure we MPI is initialized with any thread support
  init_mpi();

  // Get status of MPI before PETSc initialisation
  const bool mpi_init_status = mpi_initialized();

  // Print message if PETSc is initialised with command line arguments
  if (argc > 1)
    log(TRACE, "Initializing PETSc with given command-line arguments.");

  PetscBool is_initialized;
  PetscInitialized(&is_initialized);
  if (!is_initialized)
  {
    // Initialize PETSc
    PetscInitializeNoArguments();
  }

  // Pass command line arguments to PETSc (will overwrite any
  // default above)
  #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR <= 6 && PETSC_VERSION_RELEASE == 1
  PetscOptionsInsert(&argc, &argv, NULL);
  #else
  PetscOptionsInsert(NULL, &argc, &argv, NULL);
  #endif

  #ifdef HAS_SLEPC
  // Initialize SLEPc
  SlepcInitialize(&argc, &argv, NULL, NULL);
  #endif

  // Avoid using default PETSc signal handler
  const bool use_petsc_signal_handler = parameters["use_petsc_signal_handler"];
  if (!use_petsc_signal_handler)
    PetscPopSignalHandler();

  // Use our own error handler so we can pretty print errors from PETSc
  PetscPushErrorHandler(PetscDolfinErrorHandler, nullptr);

  // Remember that PETSc has been initialized
  singleton().petsc_initialized = true;

  // Determine if PETSc initialised MPI (and is therefore responsible
  // for MPI finalization)
  if (mpi_initialized() && !mpi_init_status)
    singleton().control_mpi = false;

  #else
  dolfin_error("SubSystemsManager.cpp",
               "initialize PETSc subsystem",
               "DOLFIN has not been configured with PETSc support");
  #endif
}
//-----------------------------------------------------------------------------
void SubSystemsManager::finalize()
{
  // Finalize subsystems in the correct order
  finalize_petsc();
  finalize_mpi();
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
  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);

  // Finalise MPI if required
  if (mpi_initialized && singleton().control_mpi)
  {
    // Check in MPI has already been finalised (possibly incorrectly by a
    // 3rd party library). If it hasn't, finalise as normal.
    int mpi_finalized;
    MPI_Finalized(&mpi_finalized);
    if (!mpi_finalized)
      MPI_Finalize();
    else
    {
      // Use std::cout since log system may fail because MPI has been shut down.
      std::cout << "DOLFIN is responsible for MPI, but it has been finalized elsewhere prematurely." << std::endl;
      std::cout << "This is usually due to a bug in a 3rd party library, and can lead to unpredictable behaviour."
                << std::endl;
      std::cout << "If using PyTrilinos, make sure that PyTrilinos modules are imported before the DOLFIN module."
                << std::endl;
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
    if (!PetscFinalizeCalled)
    {
      PetscFinalize();
    }
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
  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);
  return mpi_initialized;
  #else
  // DOLFIN is not configured for MPI (it might be through PETSc)
  return false;
  #endif
}
//-----------------------------------------------------------------------------
bool SubSystemsManager::mpi_finalized()
{
  #ifdef HAS_MPI
  int mpi_finalized;
  MPI_Finalized(&mpi_finalized);
  return mpi_finalized;
  #else
  // DOLFIN is not configured for MPI (it might be through PETSc)
  return false;
  #endif
}
//-----------------------------------------------------------------------------
#ifdef HAS_PETSC
PetscErrorCode SubSystemsManager::PetscDolfinErrorHandler(
  MPI_Comm comm, int line, const char *fun, const char *file,
  PetscErrorCode n, PetscErrorType p, const char *mess, void *ctx)
{
  // Store message for printing later (by PETScObject::petsc_error)
  // only if it's not empty message (passed by PETSc when repeating error)
  std::string _mess = mess;
  boost::algorithm::trim(_mess);
  if (_mess != "")
    singleton().petsc_err_msg = _mess;

  // Fetch PETSc error description
  const char* desc;
  PetscErrorMessage(n, &desc, nullptr);

  // Log detailed error info
  log(TRACE,
      "PetscDolfinErrorHandler: line '%d', function '%s', file '%s',\n"
      "                       : error code '%d' (%s), message follows:",
      line, fun, file, n, desc);
  // NOTE: don't put _mess as variadic argument; it might get trimmed
  log(TRACE, std::string(78, '-'));
  log(TRACE, _mess);
  log(TRACE, std::string(78, '-'));

  // Continue with error handling
  PetscFunctionReturn(n);
}
#endif
//-----------------------------------------------------------------------------
