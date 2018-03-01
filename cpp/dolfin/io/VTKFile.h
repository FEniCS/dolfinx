// Copyright (C) 2005-2017 Garth N. Wells
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
class GenericFunction;
}

namespace mesh
{
class Mesh;
template <typename T>
class MeshFunction;
}

namespace io
{

/// Output of meshes and functions in VTK format

/// XML format for visualisation purposes. It is not suitable to
/// checkpointing as it may decimate some data.

class VTKFile
{
public:
  /// Create VTK file
  VTKFile(const std::string filename);

  // Destructor
  ~VTKFile();

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
  void write_function(const function::Function& u, double time);

  void write_mesh(const mesh::Mesh& mesh, double time);

  std::string init(const mesh::Mesh& mesh, std::size_t dim) const;

  void finalize(std::string vtu_filename, double time);

  void results_write(const function::Function& u, std::string file) const;

  void write_point_data(const function::GenericFunction& u,
                        const mesh::Mesh& mesh, std::string file) const;

  void pvd_file_write(std::size_t step, double time, std::string file);

  void pvtu_write_function(std::size_t dim, std::size_t rank,
                           const std::string data_location,
                           const std::string name, const std::string filename,
                           std::size_t num_processes) const;

  void pvtu_write_mesh(const std::string pvtu_filename,
                       const std::size_t num_processes) const;

  void pvtu_write(const function::Function& u,
                  const std::string pvtu_filename) const;

  void vtk_header_open(std::size_t num_vertices, std::size_t num_cells,
                       std::string file) const;

  void vtk_header_close(std::string file) const;

  std::string vtu_name(const int process, const int num_processes,
                       const int counter, std::string ext) const;

  void clear_file(std::string file) const;

  template <typename T>
  void mesh_function_write(T& meshfunction, double time);

  // Strip path from file
  std::string strip_path(std::string file) const;

private:
  const std::string _filename;

  // Counters for the number of times various data has been written
  std::size_t counter;

  void pvtu_write_mesh(pugi::xml_node xml_node) const;
};
}
}