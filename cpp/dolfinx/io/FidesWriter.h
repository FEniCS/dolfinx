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
// Output of meshes and functions compatible with the FIDES Paraview reader
// https://fides.readthedocs.io/en/latest/paraview/paraview.html
class FidesWriter
{

public:
  /// Create Fides writer for a mesh
  /// @param[in] comm The MPI communciator
  /// @param[in] filename Name of output file
  /// @param[in] mesh The mesh
  FidesWriter(MPI_Comm comm, const std::string& filename,
              std::shared_ptr<const mesh::Mesh> mesh);

  /// Create Fides writer for list of functions (real)
  /// @param[in] comm The MPI communciator
  /// @param[in] filename Name of output file
  /// @param[in] functions List of functions
  FidesWriter(
      MPI_Comm comm, const std::string& filename,
      const std::vector<std::reference_wrapper<const fem::Function<double>>>&
          functions);

  /// Create Fides writer for list of functions (complex)
  /// @param[in] comm The MPI communciator
  /// @param[in] filename Name of output file
  /// @param[in] functions List of functions
  FidesWriter(MPI_Comm comm, const std::string& filename,
              const std::vector<std::reference_wrapper<
                  const fem::Function<std::complex<double>>>>& functions);

  /// Move constructor
  FidesWriter(FidesWriter&& file) = default;

  /// Destructor
  virtual ~FidesWriter();

  /// Close the file
  void close();

  /// Write data to file
  void write(double t);

private:
  std::unique_ptr<adios2::ADIOS> _adios;
  std::unique_ptr<adios2::IO> _io;
  std::unique_ptr<adios2::Engine> _engine;

  std::shared_ptr<const dolfinx::mesh::Mesh> _mesh;
  std::vector<std::reference_wrapper<const fem::Function<double>>> _functions;
  std::vector<std::reference_wrapper<const fem::Function<std::complex<double>>>>
      _complex_functions;
};

} // namespace dolfinx::io

#endif
