// Copyright (C) 2007 Magnus Vikstrøm.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2008.
// Modified by Ola Skavhaug, 2008-2009.
// Modified by Anders Logg, 2008-2009.
//
// First added:  2007-11-30
// Last changed: 2009-05-25

#ifndef __MPI_DOLFIN_WRAPPER_H
#define __MPI_DOLFIN_WRAPPER_H

#include <vector>
#include <dolfin/common/types.h>

#ifdef HAS_MPI
#include <mpi.h>
#endif

namespace dolfin
{

#ifdef HAS_MPI
  class MPICommunicator
  {
  public:
    /// Create communicator (copy of MPI_COMM_WORLD)
    MPICommunicator();

    /// Destructor
    ~MPICommunicator();

    /// Dereference operator
    MPI_Comm& operator*();

  private:
    MPI_Comm communicator;
  };
#endif

  /// This class provides utility functions for easy communcation with MPI.

  class MPI
  {
  public:

    /// Return proccess number
    static uint process_number();

    /// Return number of processes
    static uint num_processes();
    
    /// Determine whether we should broadcast (based on current parallel policy)
    static bool is_broadcaster();

    /// Determine whether we should receive (based on current parallel policy)
    static bool is_receiver();

    // FIXME: Write documentation for this very fancy and versatile function!
    // FIXME: The mother of all MPI calls! It does everything anyone would ever need to do with MPI... :-)

    /// Distribute local arrays on all processors according to given partition
    static void distribute(std::vector<uint>& values, std::vector<uint>& partition);

    /// Distribute local arrays on all processors according to given partition
    static void distribute(std::vector<double>& values, std::vector<uint>& partition);

    /// Gather values (wrapper for MPI_Allgather)
    static void gather(std::vector<uint>& values);

    /// Find global max value (wrapper for MPI_Allredue with MPI_MAX as reduction op)
    static uint global_maximum(uint size);

    /// Send-receive and return number of received values (wrapper for MPI_Sendrecv)
    static uint send_recv(uint* send_buffer, uint send_size, uint dest,
                          uint* recv_buffer, uint recv_size, uint source);

    /// Send-receive and return number of received values (wrapper for MPI_Sendrecv)
    static uint send_recv(double* send_buffer, uint send_size, uint dest,
                          double* recv_buffer, uint recv_size, uint source);

    /// Return local range, splitting [0, N - 1] into num_processes() portions of almost equal size
    static std::pair<uint, uint> local_range(uint N);
    
    /// Return which process owns index (inverse of local_range)
    static uint index_owner(uint index, uint N);

  };

}

#endif
