// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_ADIOS2

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
  /// Initialize ADIOS
  ADIOS2File(MPI_Comm comm, const std::string& filename,
             const std::string& mode);

  /// Move constructor
  ADIOS2File(ADIOS2File&& file) = default;

  /// Destructor
  virtual ~ADIOS2File();

  void close();

  /// Write mesh to file
  /// @param[in] mesh
  void write_mesh(const mesh::Mesh& mesh);

private:
  std::unique_ptr<adios2::ADIOS> _adios;
  std::unique_ptr<adios2::IO> _io;
  std::unique_ptr<adios2::Engine> _engine;
};

} // namespace dolfinx::io

#endif
