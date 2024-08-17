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

namespace
{
std::map<std::string, adios2::Mode> string_to_mode{
    {"write", adios2::Mode::Write},
    {"read", adios2::Mode::Read},
    {"append", adios2::Mode::Append},
};
} // namespace

namespace dolfinx::io
{

/// ADIOS2-based writers/readers
class ADIOS2Wrapper
{
public:
  /// @brief Create an ADIOS2-based engine writer/reader
  /// @param[in] comm The MPI communicator
  /// @param[in] filename Name of output file
  /// @param[in] tag The ADIOS2 IO name
  /// @param[in] engine_type ADIOS2 engine type. See
  /// https://adios2.readthedocs.io/en/latest/engines/engines.html.
  /// @param[in] mode ADIOS2 mode, default is Write or Read
  ADIOS2Wrapper(MPI_Comm comm, std::string filename, std::string tag,
                std::string engine_type = "BP5", std::string mode = "write")
      : _adios(std::make_shared<adios2::ADIOS>(comm)),
        _io(std::make_shared<adios2::IO>(_adios->DeclareIO(tag)))
  {
    _io->SetEngine(engine_type);
    _engine = std::make_shared<adios2::Engine>(
        _io->Open(filename, string_to_mode[mode]));
  }

  /// @brief Create an ADIOS2-based engine writer/reader
  /// @param[in] config_file Path to config file to set up ADIOS2 engine from
  /// @param[in] comm The MPI communicator
  /// @param[in] filename Name of output file
  /// @param[in] tag The ADIOS2 IO name
  /// @param[in] mode ADIOS2 mode, default is Write or Read
  ADIOS2Wrapper(std::string config_file, MPI_Comm comm, std::string filename,
                std::string tag, std::string mode = "append")
  {
    _adios = std::make_shared<adios2::ADIOS>(config_file, comm);
    _io = std::make_shared<adios2::IO>(_adios->DeclareIO(tag));
    _engine = std::make_shared<adios2::Engine>(
        _io->Open(filename, string_to_mode[mode]));
  }

  /// @brief Move constructor
  ADIOS2Wrapper(ADIOS2Wrapper&& ADIOS2) = default;

  /// @brief Copy constructor
  ADIOS2Wrapper(const ADIOS2Wrapper&) = delete;

  /// @brief Destructor
  ~ADIOS2Wrapper() { close(); }

  /// @brief Move assignment
  ADIOS2Wrapper& operator=(ADIOS2Wrapper&& ADIOS2) = default;

  // Copy assignment
  ADIOS2Wrapper& operator=(const ADIOS2Wrapper&) = delete;

  /// @brief  Close the file
  void close()
  {
    assert(_engine);
    if (*_engine)
      _engine->Close();
  }

  /// @brief  Get the IO object
  std::shared_ptr<adios2::IO> io() { return _io; }

  /// @brief  Get the Engine object
  std::shared_ptr<adios2::Engine> engine() { return _engine; }

private:
  std::shared_ptr<adios2::ADIOS> _adios;
  std::shared_ptr<adios2::IO> _io;
  std::shared_ptr<adios2::Engine> _engine;
};

} // namespace dolfinx::io

#endif
