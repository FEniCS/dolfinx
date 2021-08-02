// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_ADIOS2

#include "utils.h"
#include <array>
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
}

namespace dolfinx::io
{

class FidesWriter
{
public:
  /// Create Fides writer for a mesh
  FidesWriter(MPI_Comm comm, const std::string& filename, io::mode mode,
              std::shared_ptr<const mesh::Mesh> mesh);

  /// Create Fides writer for list of functions
  FidesWriter(
      MPI_Comm comm, const std::string& filename, io::mode mode,
      const std::vector<std::reference_wrapper<const fem::Function<double>>>&
          functions);

  FidesWriter(MPI_Comm comm, const std::string& filename, io::mode mode,
              const std::vector<std::reference_wrapper<
                  const fem::Function<std::complex<double>>>>& functions);

  /// Move constructor
  FidesWriter(FidesWriter&& file) = default;

  /// Destructor
  virtual ~FidesWriter();

  /// Close the file
  void close();

  // /// Write mesh to file
  // /// @param[in] mesh
  // void write_mesh(const mesh::Mesh& mesh);

  // /// Write an arrays of Functions to file
  // /// @param[in] u Functions to write
  // void write_function(
  //     const std::vector<std::reference_wrapper<const fem::Function<double>>>&
  //         u);

  // /// Write list of functions to file
  // /// @param[in] function
  // void write_function(const std::vector<std::reference_wrapper<
  //                         const fem::Function<std::complex<double>>>>& u);

  void write(double t);

private:
  std::unique_ptr<adios2::ADIOS> _adios;
  std::unique_ptr<adios2::IO> _io;
  std::unique_ptr<adios2::Engine> _engine;

  std::shared_ptr<const dolfinx::mesh::Mesh> _mesh;
  std::vector<std::reference_wrapper<const fem::Function<double>>> _functions;
  std::vector<std::reference_wrapper<const fem::Function<std::complex<double>>>>
      _complex_functions;
  bool _mesh_written;
};

} // namespace dolfinx::io

#endif
