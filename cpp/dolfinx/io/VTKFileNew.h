// Copyright (C) 2020 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "pugixml.hpp"
#include <dolfinx/common/MPI.h>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

// namespace pugi
// {
// class xml_node;
// }

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

/// Output of meshes and functions in VTK format

/// XML format is suitable for visualisation of higher order geometries.
/// It is not suitable to checkpointing as it may decimate some data.

class VTKFileNew
{
public:
  /// Create VTK file
  VTKFileNew(MPI_Comm comm, const std::string filename,
             const std::string file_mode);

  /// Destructor
  ~VTKFileNew();

  /// Write mesh to file. Supports arbitrary order Lagrange
  /// isoparametric cells.
  /// @param[in] mesh The mesh to write to file
  /// @param[in] time Time parameter to associate with the @p mesh
  void write(const mesh::Mesh& mesh, double time = 0.0);

  /// Output function::Function
  // void write(const function::Function& u);

  // /// Output function::Function and timestep
  // void write(const function::Function& u, double t);

private:
  pugi::xml_document _pvd_xml;

  std::string _filename;

  // MPI communicator
  dolfinx::MPI::Comm _comm;

  // Counter for the number of times various data has been written
  // std::size_t _counter;
};
} // namespace io
} // namespace dolfinx
