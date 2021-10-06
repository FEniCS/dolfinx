// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_ADIOS2

#include "FidesWriter.h"
#include "utils.h"
#include <array>
#include <dolfinx/common/MPI.h>
#include <memory>
#include <string>
#include <variant>
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
class VTXWriter : public ADIOS2Writer
{
public:
  /// Createa VTX writer for a mesh
  /// @param[in] comm The MPI communicator
  /// @param[in] filename Name of output file
  /// @param[in] mesh The mesh to write
  VTXWriter(MPI_Comm comm, const std::string& filename,
            std::shared_ptr<const mesh::Mesh> mesh);

  /// Create a VTX writer for list of functions
  /// @param[in] comm The MPI communicator
  /// @param[in] filename Name of output file
  /// @param[in] functions List of functions to write
  VTXWriter(
      MPI_Comm comm, const std::string& filename,
      const std::vector<std::variant<std::shared_ptr<const ADIOS2Writer::U0>,
                                     std::shared_ptr<const ADIOS2Writer::U1>>>&
          u);

  /// Move constructor
  VTXWriter(VTXWriter&& file) = default;

  /// Destructor
  ~VTXWriter() = default;

  /// Write data to file
  /// @param[in] t The time step
  void write(double t);

private:
  // Flag to indicate if mesh should be written, or if mesh is defined
  // through dof coordinates
  bool _write_mesh_data;
};

} // namespace dolfinx::io

#endif
