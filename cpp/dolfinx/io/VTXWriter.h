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
} // namespace dolfinx::mesh

namespace dolfinx::io
{
// Writer for meshes and functions using ADIOS2 VTX format
// https://adios2.readthedocs.io/en/latest/ecosystem/visualization.html#using-vtk-and-paraview
class VTXWriter
{
public:
  /// Create VTX writer for a mesh
  /// @param[in] comm The MPI communciator
  /// @param[in] filename Name of output file
  /// @param[in] mode The file mode (read/write/append)
  /// @param[in] mesh The mesh
  VTXWriter(MPI_Comm comm, const std::string& filename, io::mode mode,
            std::shared_ptr<const mesh::Mesh> mesh);

  /// Create VTX writer for list of functions (real)
  /// @param[in] comm The MPI communciator
  /// @param[in] filename Name of output file
  /// @param[in] mode The file mode (read/write/append)
  /// @param[in] functions List of functions
  VTXWriter(
      MPI_Comm comm, const std::string& filename, io::mode mode,
      const std::vector<std::reference_wrapper<const fem::Function<double>>>&
          functions);

  /// Create VTX writer for list of functions (complex)
  /// @param[in] comm The MPI communciator
  /// @param[in] filename Name of output file
  /// @param[in] mode The file mode (read/write/append)
  /// @param[in] functions List of functions
  VTXWriter(MPI_Comm comm, const std::string& filename, io::mode mode,
            const std::vector<std::reference_wrapper<
                const fem::Function<std::complex<double>>>>& functions);

  /// Move constructor
  VTXWriter(VTXWriter&& file) = default;

  /// Destructor
  virtual ~VTXWriter();

  /// Close the file
  void close();

  /// Write data to file
  /// @param[in] t The time step
  void write(double t);

private:
  std::unique_ptr<adios2::ADIOS> _adios;
  std::unique_ptr<adios2::IO> _io;
  std::unique_ptr<adios2::Engine> _engine;

  std::shared_ptr<const dolfinx::mesh::Mesh> _mesh;
  std::vector<std::reference_wrapper<const fem::Function<double>>> _functions;
  std::vector<std::reference_wrapper<const fem::Function<std::complex<double>>>>
      _complex_functions;
  // Flag to indicate if mesh should be written, or if mesh is defined through
  // dof coordinates
  bool _write_mesh_data;
};

} // namespace dolfinx::io

#endif
