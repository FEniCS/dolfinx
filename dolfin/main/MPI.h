// Copyright (C) 2007 Magnus Vikstrøm.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2008.
// Modified by Ola Skavhaug, 2008.
// Modified by Anders Logg, 2008.
//
// First added:  2007-11-30
// Last changed: 2008-09-18

#ifndef __MPI_helper_H
#define __MPI_helper_H

#include <dolfin/common/types.h>

namespace dolfin
{

  /// This class provides utility functions for easy communcation with MPI.
  
  class MPI
  {
  public:

    /// Return proccess number
    static uint process_number();

    /// Return number of processes
    static uint num_processes();

    /// Determine whether we should broadcast (based on current parallel policy)
    static bool broadcast();

    /// Determine whether we should receive (based on current parallel policy)
    static bool receive();

  };
}

#endif
