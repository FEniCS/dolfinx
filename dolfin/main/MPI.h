// Copyright (C) 2007 Magnus Vikstrøm.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2008.
// Modified by Ola Skavhaug, 2008.
//
// First added:  2007-11-30
// Last changed: 2008-08-13

#ifndef __MPI_helper_H
#define __MPI_helper_H

#include <dolfin/common/types.h>

namespace dolfin
{
  /// This class provides utility functions for easy access of the number of 
  /// processes and current process number.
  
  class MPI
  {
  public:

    /// Return proccess number
    static uint processNumber();

    /// Return number of processes
    static uint numProcesses();

    /// Determine whether we should broadcast (based on current parallel policy)
    static bool broadcast();

    /// Determine whether we should receive (based on current parallel policy)
    static bool receive();

  };
}

#endif
