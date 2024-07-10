// Copyright (C) 2024 Abdullah Mujahid
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_ADIOS2

#include "ADIOS2_utils.h"
#include <adios2.h>
#include <basix/finite-element.h>
#include <dolfinx/mesh/Mesh.h>
#include <mpi.h>

/// @file checkpointing.h
/// @brief ADIOS2 based checkpointing

namespace dolfinx::io::checkpointing
{

  /// @brief Write mesh to a file.
  ///
  /// @param[in] comm The MPI communicator to open the file on
  /// @param[in] filename Name of output file
  /// @param[in] tag ADIOS2 tag for IO
  /// @param[in] mesh Mesh of type float to write to the file
  /// @note This is experimental version. Expected would be to
  /// pass ADIOS2Engine object.
void write(MPI_Comm comm, std::string filename, std::string tag,
           //    adios2::IO io, adios2::Engine engine,
           std::shared_ptr<dolfinx::mesh::Mesh<float>> mesh);

  /// @brief Write mesh to a file.
  ///
  /// @param[in] comm The MPI communicator to open the file on
  /// @param[in] filename Name of output file
  /// @param[in] tag ADIOS2 tag for IO
  /// @param[in] mesh Mesh of type double to write to the file
  /// @note This is experimental version. Expected would be to
  /// pass ADIOS2Engine object.
void write(MPI_Comm comm, std::string filename, std::string tag,
           //    adios2::IO io, adios2::Engine engine,
           std::shared_ptr<dolfinx::mesh::Mesh<double>> mesh);

} // namespace dolfinx::io::checkpointing

#endif
