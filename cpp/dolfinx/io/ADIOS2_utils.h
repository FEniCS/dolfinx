// Copyright (C) 2021-2023 Jørgen S. Dokken and Garth N. Wells
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
class ADIOS2Engine
{
public:
  /// @brief Create an ADIOS2-based engine writer/reader
  /// @param[in] comm The MPI communicator
  /// @param[in] filename Name of output file
  /// @param[in] tag The ADIOS2 object name
  /// @param[in] engine ADIOS2 engine type. See
  /// https://adios2.readthedocs.io/en/latest/engines/engines.html.
  /// @param[in] mode ADIOS2 mode, default is Write or Read
  ADIOS2Engine(MPI_Comm comm, const std::filesystem::path& filename,
               std::string tag, std::string engine = "BP5",
               const adios2::Mode mode = adios2::Mode::Write);

  /// @brief Move constructor
  ADIOS2Engine(ADIOS2Engine&& engine) = default;

  /// @brief Copy constructor
  ADIOS2Engine(const ADIOS2Engine&) = delete;

  /// @brief Destructor
  ~ADIOS2Engine();

  /// @brief Move assignment
  ADIOS2Engine& operator=(ADIOS2Engine&& engine) = default;

  // Copy assignment
  ADIOS2Engine& operator=(const ADIOS2Engine&) = delete;

  /// @brief  Close the file
  void close();

  // /// @brief  Get the IO object
  // std::unique_ptr<adios2::IO> io();

  // /// @brief  Close the Engine object
  // std::unique_ptr<adios2::Engine> engine();

protected:
  std::unique_ptr<adios2::ADIOS> _adios;
  std::unique_ptr<adios2::IO> _io;
  std::unique_ptr<adios2::Engine> _engine;
};

} // namespace dolfinx::io

#endif
