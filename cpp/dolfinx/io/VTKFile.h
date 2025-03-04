// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <concepts>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/types.h>
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
template <dolfinx::scalar T, std::floating_point U>
class Function;
}

namespace dolfinx::mesh
{
template <std::floating_point T>
class Mesh;
}

namespace dolfinx::io
{

/// @brief Output of meshes and functions in VTK/ParaView format.
///
/// Isoparametric meshes of arbitrary degree are supported. For finite
/// element functions, cell-based (DG0) and Lagrange (point-based)
/// functions can be saved. For vertex-based functions the output must
/// be isoparametic, i.e. the geometry and the finite element functions
/// must be defined using the same basis.
///
/// @warning This format is not suitable for checkpointing.
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

  /// @brief Write a mesh to file. Supports arbitrary order Lagrange
  /// isoparametric cells.
  /// @param[in] mesh Mesh to write to file.
  /// @param[in] time Time parameter to associate with `mesh`.
  template <std::floating_point U>
  void write(const mesh::Mesh<U>& mesh, double time = 0.0);

  /// @brief Write finite elements function with an associated time
  /// step.
  ///
  /// @pre Functions in `u` cannot be sub-Functions. Extract
  /// sub-Functions before output.
  ///
  /// @pre All Functions in `u` with point-wise data must use the same
  /// element type (up to the block size) and the element must be
  /// (discontinuous) Lagrange. Interpolate fem::Function before output
  /// if required.
  ///
  /// @param[in] u List of functions to write to file
  /// @param[in] t Time parameter to associate with @p u
  /// @pre All Functions in `u` must share the same mesh
  template <dolfinx::scalar T, std::floating_point U = scalar_value_t<T>>
  void
  write(const std::vector<std::reference_wrapper<const fem::Function<T, U>>>& u,
        double t);

private:
  std::unique_ptr<pugi::xml_document> _pvd_xml;

  std::filesystem::path _filename;

  // MPI communicator
  dolfinx::MPI::Comm _comm;
};
} // namespace dolfinx::io
