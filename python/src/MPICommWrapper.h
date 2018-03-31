// Copyright (C) 2017 Tormod Landet
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

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
