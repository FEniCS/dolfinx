// Copyright (C) 2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-01-07
// Last changed: 

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
    static void initMPI();

    /// Initialize PETSc without command-line arguments
    static void initPETSc();

    /// Initialize PETSc with command-line arguments
    static void initPETSc(int argc, char* argv[], bool cmd_line_args = true);

  private:

    // Constructor
    SubSystemsManager();

    // Copy construtor
    SubSystemsManager(const SubSystemsManager& sub_sys_manager);

    // Destructor
    ~SubSystemsManager();

    /// Finalize MPI
    static void finalizeMPI();

    /// Finalize PETSc
    static void finalizePETSc();

    // Check if MPI has been initialised
    static bool MPIinitialized();

    // Check if MPI has been initialised
    static void MPIinit();

    static SubSystemsManager sub_systems_manager;

    bool petsc_initialized;
    bool petsc_controls_mpi;
  };

}

#endif
