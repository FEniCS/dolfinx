// Copyright (C) 2005-2019 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <fstream>
#include <petscsys.h>
#include <string>
#include <utility>
#include <vector>

namespace pugi
{
class xml_node;
}

namespace dolfinx
{
namespace function
{
template <typename T>
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

class VTKFile
{
public:
  /// Create VTK file
  VTKFile(const std::string filename);

  /// Destructor
  ~VTKFile() = default;

  /// Output mesh
  void write(const mesh::Mesh& mesh);

  /// Output function::Function
  void write(const function::Function<PetscScalar>& u);

  /// Output mesh::Mesh and timestep
  void write(const mesh::Mesh& mesh, double t);

  /// Output function::Function and timestep
  void write(const function::Function<PetscScalar>& u, double t);

private:
  const std::string _filename;

  // Counter for the number of times various data has been written
  std::size_t _counter;
};
} // namespace io
} // namespace dolfinx
