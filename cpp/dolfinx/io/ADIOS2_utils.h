// Copyright (C) 2024 Abdullah Mujahid, Jørgen S. Dokken and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_ADIOS2

#include <adios2.h>
#include <cassert>
#include <filesystem>
#include <mpi.h>

/// @file ADIOS2_utils.h
/// @brief Utils for ADIOS2

namespace dolfinx::io
{

/// ADIOS2-based writers/readers
class ADIOS2Container
{
public:
  /// @brief Create an ADIOS2-based engine writer/reader
  /// @param[in] comm The MPI communicator
  /// @param[in] filename Name of output file
  /// @param[in] tag The ADIOS2 object name
  /// @param[in] engine ADIOS2 engine type. See
  /// https://adios2.readthedocs.io/en/latest/engines/engines.html.
  /// @param[in] mode ADIOS2 mode, default is Write or Read
  ADIOS2Container(MPI_Comm comm, std::string filename,
               std::string tag, std::string engine = "BP5",
               std::string mode = "write");

  /// @brief Move constructor
  ADIOS2Container(ADIOS2Container&& engine) = default;

  /// @brief Copy constructor
  ADIOS2Container(const ADIOS2Container&) = delete;

  /// @brief Destructor
  ~ADIOS2Container();

  /// @brief Move assignment
  ADIOS2Container& operator=(ADIOS2Container&& engine) = default;

  // Copy assignment
  ADIOS2Container& operator=(const ADIOS2Container&) = delete;

  /// @brief  Close the file
  void close();

  /// @brief  Get the IO object
  std::shared_ptr<adios2::IO> io() { return _io; }

  /// @brief  Close the Engine object
  std::shared_ptr<adios2::Engine> engine() { return _engine; }

protected:
  std::shared_ptr<adios2::ADIOS> _adios;
  std::shared_ptr<adios2::IO> _io;
  std::shared_ptr<adios2::Engine> _engine;
};

} // namespace dolfinx::io

#endif