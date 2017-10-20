// Copyright (C) 2017 Tormod Landet
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

#ifndef __MPI_COMM_WRAPPER_H
#define __MPI_COMM_WRAPPER_H

#include <dolfin/common/MPI.h>

namespace dolfin_wrappers
{

  /// This class wraps the MPI_Comm type for use in the pybind11
  /// generation of python wrappers. MPI_Comm is either a pointer or
  /// an int (MPICH vs OpenMPI) and this cannot be wrapped in a type
  /// safe way with pybind11.

  class MPICommWrapper
  {
  public:

    MPICommWrapper();

    /// Wrap a MPI_Comm object
    MPICommWrapper(MPI_Comm comm);

    /// Assignment operator
    MPICommWrapper& operator=(const MPI_Comm comm);

    /// Get the underlying MPI communicator
    MPI_Comm get() const;

  private:

    // The underlying communicator
    MPI_Comm _comm;

  };
}

#endif
