// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/MPI.h>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

namespace pugi
{
class xml_document;
}

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

/// Output of meshes and functions in VTK/ParaView format. Isoparametric
/// meshes of arbitrary degree are supported. For finite element
/// functions, cell-based (DG0) and Lagrange (point-based) functions can
/// be saved. For vertex-based functions the output must be
/// isoparametic, i.e. the geometry and the finite element functions
/// must be defined using the same basis.
///
/// @warning This format is not suitable to checkpointing
class VTKFile
{
public:
  /// Create VTK file
  VTKFile(MPI_Comm comm, const std::filesystem::path& filename,
          const std::string& file_mode);

  /// Destructor
  ~VTKFile();

  /// Close file
  void close();

  /// Flushes XML files to disk
  void flush();

  /// Write a mesh to file. Supports arbitrary order Lagrange
  /// isoparametric cells.
  /// @param[in] mesh The Mesh to write to file
  /// @param[in] time Time parameter to associate with @p mesh
  void write(const mesh::Mesh& mesh, double time = 0.0);

  /// Write a finite element function with an associated timestep
  /// @param[in] u List of functions to write to file
  /// @param[in] t Time parameter to associate with @p u
  void write(
      const std::vector<std::reference_wrapper<const fem::Function<double>>>& u,
      double t);

  /// Write a finite element function with an associated timestep
  /// @param[in] u List of functions to write to file
  /// @param[in] t Time parameter to associate with @p u
  void write(
      const std::vector<
          std::reference_wrapper<const fem::Function<std::complex<double>>>>& u,
      double t);

private:
  std::unique_ptr<pugi::xml_document> _pvd_xml;

  std::filesystem::path _filename;

  // MPI communicator
  dolfinx::MPI::Comm _comm;
};
} // namespace dolfinx::io
