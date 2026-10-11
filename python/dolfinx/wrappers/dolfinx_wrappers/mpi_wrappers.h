// Copyright (C) 2017-2026 Tormod Landet and Garth N. Wells
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
  explicit MPICommWrapper(MPI_Comm comm) : _comm(comm) {}

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

/// This class wraps the MPI_Request type for use in the nanobind
/// generation of python wrappers. MPI_Request is either a pointer or
/// an int (MPICH vs OpenMPI) and this cannot be wrapped in a type safe
/// way with nanobind.

class MPIRequestWrapper
{
public:
  MPIRequestWrapper() : _request(MPI_REQUEST_NULL) {}

  /// Wrap an MPI_Request object
  explicit MPIRequestWrapper(MPI_Request request) : _request(request) {}

  /// Assignment operator
  MPIRequestWrapper& operator=(const MPI_Request request)
  {
    this->_request = request;
    return *this;
  }

  /// Get the underlying MPI request
  MPI_Request get() const { return _request; }

private:
  // The underlying request
  MPI_Request _request;
};
} // namespace dolfinx_wrappers
