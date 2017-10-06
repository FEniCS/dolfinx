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

namespace dolfin
{

  /// This class wraps the MPI_Comm type for use in the pybind11
  /// generation of python wrappers. MPI_Comm is either a pointer
  /// or an int (MPICH vs OpenMPI) and this cannot be wrapped in a
  /// type safe way with pybind11. This class is NOT used from dolfin
  /// C++ side, the only reason it is here instead of in the 
  /// dolfin_wrappers namespace is to make it available to code being
  /// jit compiled from python with dolfin.compile_cpp_code(...)

  class MPICommWrapper {

    MPI_Comm comm;

  public:

    MPICommWrapper() {};

    MPICommWrapper(MPI_Comm comm);

    MPICommWrapper &operator=(const MPI_Comm comm);

    MPI_Comm get() const;

  };
}

#endif

