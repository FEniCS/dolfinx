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
    {"readrandomaccess", adios2::Mode::ReadRandomAccess},
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
  ADIOS2Wrapper(MPI_Comm comm, std::string filename)
      : _filename(filename), _adios(std::make_shared<adios2::ADIOS>(comm))
  {
  }

  /// @brief Create an ADIOS2-based engine writer/reader
  /// @param[in] comm The MPI communicator
  /// @param[in] filename Name of output file
  /// @param[in] tag The ADIOS2 IO name
  /// @param[in] engine_type ADIOS2 engine type. See
  /// https://adios2.readthedocs.io/en/latest/engines/engines.html.
  /// @param[in] mode ADIOS2 mode, default is Write or Read
  ADIOS2Wrapper(MPI_Comm comm, std::string filename, std::string tag,
                std::string engine_type = "BP5", std::string mode = "write")
      : _filename(filename), _adios(std::make_shared<adios2::ADIOS>(comm))
  {
    _ios.insert({tag, std::make_shared<adios2::IO>(_adios->DeclareIO(tag))});
    _ios[tag]->SetEngine(engine_type);
    _engines.insert(tag, std::make_shared<adios2::Engine>(
                             _ios[tag]->Open(filename, string_to_mode[mode])));
  }

  /// @brief Create an ADIOS2-based engine writer/reader
  /// @param[in] config_file Path to config file to set up ADIOS2 engines from
  /// @param[in] comm The MPI communicator
  /// @param[in] filename Name of output file
  /// @param[in] tag The ADIOS2 IO name
  /// @param[in] mode ADIOS2 mode, default is Write or Read
  ADIOS2Wrapper(std::string config_file, MPI_Comm comm, std::string filename,
                std::string tag, std::string mode = "append")
      : _filename(filename)
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

  /// @brief  Add IO and an Engine with specified type and mode
  /// @param[in] tag The ADIOS2 IO name
  /// @param[in] engine_type ADIOS2 engine type. See
  /// https://adios2.readthedocs.io/en/latest/engines/engines.html.
  /// @param[in] mode ADIOS2 mode, available are: "write", "read", "append",
  /// "readrandomaccess", and the default is "append"
  void add_io(std::string tag, std::string engine_type = "BP5",
              std::string mode = "append")
  {
    std::map<std::string, std::shared_ptr<adios2::IO>>::iterator it_io
        = _ios.end();
    _ios.insert(it_io,
                std::pair<std::string, std::shared_ptr<adios2::IO>>(
                    tag, std::make_shared<adios2::IO>(_adios->DeclareIO(tag))));

    assert(_ios[tag]);
    _ios[tag]->SetEngine(engine_type);
    std::map<std::string, std::shared_ptr<adios2::Engine>>::iterator it_engine
        = _engines.end();
    _engines.insert(it_engine,
                    std::pair<std::string, std::shared_ptr<adios2::Engine>>(
                        tag, std::make_shared<adios2::Engine>(_ios[tag]->Open(
                                 _filename, string_to_mode[mode]))));
  }

  /// @brief  Close engine associated with the IO with the given tag
  /// @param[in] tag The ADIOS2 IO name whose associated engine needs to be
  /// closed
  void close(std::string tag)
  {
    assert(_engines[tag]);
    if (*_engines[tag])
      _engines[tag]->Close();
  }

  /// @brief  Close all Engines
  void close()
  {
    for (auto it = _engines.begin(); it != _engines.end(); ++it)
    {
      auto engine = it->second;
      assert(engine);
      if (*engine)
        engine->Close();
    }
  }

  /// @brief  Get the IO with the given tag
  /// @param[in] tag The ADIOS2 IO name
  std::shared_ptr<adios2::IO> io(std::string tag) { return _ios[tag]; }

  /// @brief  Get the Engine object
  /// @param[in] tag The ADIOS2 IO name
  std::shared_ptr<adios2::Engine> engine(std::string tag)
  {
    return _engines[tag];
  }

private:
  std::string _filename;
  std::shared_ptr<adios2::ADIOS> _adios;
  std::map<std::string, std::shared_ptr<adios2::IO>> _ios;
  std::map<std::string, std::shared_ptr<adios2::Engin>> _engines;
};

} // namespace dolfinx::io

#endif
