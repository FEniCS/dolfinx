// Copyright (C) 2008-2011 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-01-07
// Last changed: 2011-01-23

#ifndef __SUB_SYSTEMS_MANAGER_H
#define __SUB_SYSTEMS_MANAGER_H

namespace dolfin
{

  /// This is a singleton class which manages the initialisation and
  /// finalisation of various sub systems, such as MPI and PETSc.

  class SubSystemsManager
  {
  public:

    /// Initialise MPI
    static void init_mpi();

    /// Initialize PETSc without command-line arguments
    static void init_petsc();

    /// Initialize PETSc with command-line arguments. Note that PETSc
    /// command-line arguments may also be filtered and sent to PETSc
    /// by parameters.parse(argc, argv).
    static void init_petsc(int argc, char* argv[]);

    /// Finalize subsytems. This will be called by the destructor, but in
    /// special cases it may be necessary to call finalize() explicitly.
    static void finalize();

    /// Return true if DOLFIN intialised MPI (and is therefore responsible
    //  for finalization)
    static bool responsible_mpi();

    /// Return true if DOLFIN intialised PETSc (and is therefore responsible
    //  for finalization)
    static bool responsible_petsc();

    // Check if MPI has been initialised (returns true if MPI has been
    // initialised, even if it is later finalised)
    static bool mpi_initialized();

    // Check if MPI has been finalized (returns true if MPI has been
    // finalised)
    static bool mpi_finalized();

  private:

    // Constructor (private)
    SubSystemsManager();

    // Copy constructor (private)
    SubSystemsManager(const SubSystemsManager& sub_sys_manager);

    // Destructor
    ~SubSystemsManager();

    // Finalize MPI
    static void finalize_mpi();

    // Finalize PETSc
    static void finalize_petsc();

    // Singleton instance
    static SubSystemsManager& singleton();

    // State variables
    bool petsc_initialized;
    bool control_mpi;

  };

}

#endif
