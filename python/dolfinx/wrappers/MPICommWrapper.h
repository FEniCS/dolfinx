// Copyright (C) 2017 Tormod Landet
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/MPI.h>

namespace dolfinx_wrappers
{
/// This class wraps the MPI_Comm type for use in the nanobind
/// generation of python wrappers. MPI_Comm is either a pointer or an
/// int (MPICH vs OpenMPI) and this cannot be wrapped in a type safe way
/// with nanobind

class MPICommWrapper
{
public:
  MPICommWrapper() : _comm(MPI_COMM_NULL) {}

  /// Wrap a MPI_Comm object
  MPICommWrapper(MPI_Comm comm) : _comm(comm) {}

  /// Assignment operator
  MPICommWrapper& operator=(const MPI_Comm comm)
  {
    this->_comm = comm;
    return *this;
  }

  /// Get the underlying MPI communicator
  MPI_Comm get() const { return _comm; }

private:
  // The underlying communicator
  MPI_Comm _comm;
};
} // namespace dolfinx_wrappers
