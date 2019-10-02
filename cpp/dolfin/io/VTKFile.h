// Copyright (C) 2005-2019 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace pugi
{
class xml_node;
}

namespace dolfin
{
namespace function
{
class Function;
}

namespace mesh
{
class Mesh;
template <typename T>
class MeshFunction;
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

  /// Output mesh::MeshFunction<bool>
  void write(const mesh::MeshFunction<bool>& meshfunction);

  /// Output mesh::MeshFunction<std::size_t>
  void write(const mesh::MeshFunction<std::size_t>& meshfunction);

  /// Output mesh::MeshFunction<int>
  void write(const mesh::MeshFunction<int>& meshfunction);

  /// Output mesh::MeshFunction<double>
  void write(const mesh::MeshFunction<double>& meshfunction);

  /// Output function::Function
  void write(const function::Function& u);

  /// Output mesh::Mesh and timestep
  void write(const mesh::Mesh& mesh, double t);

  /// Output mesh::MeshFunction and timestep
  void write(const mesh::MeshFunction<int>& mesh, double t);

  /// Output mesh::MeshFunction and timestep
  void write(const mesh::MeshFunction<std::size_t>& mf, double t);

  /// Output mesh::MeshFunction and timestep
  void write(const mesh::MeshFunction<double>& mf, double t);

  /// Output mesh::MeshFunction and timestep
  void write(const mesh::MeshFunction<bool>& mf, double t);

  /// Output function::Function and timestep
  void write(const function::Function& u, double t);

private:
  const std::string _filename;

  // Counter for the number of times various data has been written
  std::size_t _counter;
};
} // namespace io
} // namespace dolfin
