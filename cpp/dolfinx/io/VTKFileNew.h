// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/MPI.h>
#include <functional>
#include <memory>
#include <string>

namespace pugi
{
class xml_document;
}

namespace dolfinx
{
namespace function
{
class Function;
}

namespace mesh
{
class Mesh;
} // namespace mesh

namespace io
{

/// Output of meshes and functions in VTK/ParaView format. Isoparametric
/// meshes of arbitrary degree are supported. For finite element
/// functions, cell-based (DG0) and Lagrange (point-based) functions can
/// be saved. For vertex-based functions the output must be
/// isoparametic, i.e. the geometry and the finite element functions
/// must be defined using the same basis.

/// This format if It is not suitable to checkpointing as it may
/// decimate some data.

class VTKFileNew
{
public:
  /// Create VTK file
  VTKFileNew(MPI_Comm comm, const std::string filename,
             const std::string file_mode);

  /// Destructor
  ~VTKFileNew();

  /// Close file
  void close();

  /// Flushes XML files to disk
  void flush();

  /// Write mesh to file. Supports arbitrary order Lagrange
  /// isoparametric cells.
  /// @param[in] mesh The Mesh to write to file
  /// @param[in] time Time parameter to associate with the @p mesh
  void write(const mesh::Mesh& mesh, double time = 0.0);

  /// Write a Function to a VTK file for visualisation
  /// @param[in] u The Function to write to file
  /// @param[in] time Time parameter to associate with the @p mesh
  void
  write(const std::vector<std::reference_wrapper<const function::Function>>& u,
        double time = 0.0);

private:
  std::unique_ptr<pugi::xml_document> _pvd_xml;

  std::string _filename;

  // MPI communicator
  dolfinx::MPI::Comm _comm;
};
} // namespace io
} // namespace dolfinx
