// Copyright (C) 2007 Magnus Vikstrøm.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-11-30
// Last changed: 2007-12-01

#ifndef __MPI_MANAGER_H
#define __MPI_MANAGER_H

#include <dolfin/MeshFunction.h>

#ifdef HAVE_MPI_H
#include <mpi.h>
#endif

namespace dolfin
{
  /// This class handles initialization/finalization of MPI and provides
  /// utility functions for easy access of the number of processes and
  /// current process number.
  
  class MPIManager
  {
  public:

    /// Initialize MPI
    static void init();

    /// Finalize MPI
    static void finalize();
    
    /// Return proccess number
    static uint processNumber();

    /// Return number of processes
    static uint numProcesses();

    /// Determine whether we should broadcast (based on current parallel policy)
    static bool broadcast();

    /// Determine whether we should receive (based on current parallel policy)
    static bool receive();

  private:

    /// Constructor
    MPIManager();

    /// Destructor
    ~MPIManager();

    /// Singleton instance of MPIManager
    static MPIManager mpi;
  
  };
}

#endif
