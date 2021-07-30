// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_ADIOS2

#include "utils.h"
#include <dolfinx/common/MPI.h>
#include <memory>
#include <string>
#include <vector>

namespace adios2
{
class ADIOS;
class IO;
class Engine;
} // namespace adios2

namespace dolfinx::fem
{
template <typename T>
class Function;
}

namespace dolfinx::mesh
{
class Mesh;
} // namespace dolfinx::mesh

namespace dolfinx::io
{

class ADIOS2File
{
public:
  /// Create and ADIOS file
  ADIOS2File(MPI_Comm comm, const std::string& filename, io::mode mode);

  /// Move constructor
  ADIOS2File(ADIOS2File&& file) = default;

  /// Destructor
  virtual ~ADIOS2File();

  /// Close the file
  void close();

  /// Write mesh to file
  /// @param[in] mesh
  void write_mesh(const mesh::Mesh& mesh);

  /// Write an arrays of Functions to file
  /// @param[in] u Functions to write
  void write_function(
      const std::vector<std::reference_wrapper<const fem::Function<double>>>&
          u);

  /// Write list of functions to file
  /// @param[in] function
  void write_function(const std::vector<std::reference_wrapper<
                          const fem::Function<std::complex<double>>>>& u);

private:
  std::unique_ptr<adios2::ADIOS> _adios;
  std::unique_ptr<adios2::IO> _io;
  std::unique_ptr<adios2::Engine> _engine;

  // List with names of functions added to file
  std::vector<std::string> _functions;

  // Corresponding list of association (cell, point)
  std::vector<std::string> _associations;
};

} // namespace dolfinx::io

#endif
