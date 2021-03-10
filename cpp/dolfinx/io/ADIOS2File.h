// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_ADIOS2

#include <dolfinx/common/MPI.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <memory>
#include <vector>

namespace adios2
{
class ADIOS;
class IO;
class Engine;
} // namespace adios2

namespace dolfinx
{

namespace io
{
class ADIOS2File
{
public:
  /// Initialize ADIOS
  ADIOS2File(MPI_Comm comm, std::string filename, std::string mode);

  /// Destructor
  ~ADIOS2File();

  void close();

  /// Write a list of functions to file (real mode)
  /// @param[in] u - List of functions
  /// @param[in] t - The time step
  void write_function(
      const std::vector<std::reference_wrapper<const fem::Function<double>>>& u,
      double t = 0.0);

  // Write a list of functions to file (complex mode)
  /// @param[in] u - List of functions
  /// @param[in] t - The time step
  void write_function(
      const std::vector<
          std::reference_wrapper<const fem::Function<std::complex<double>>>>& u,
      double t = 0.0);

  /// Write mesh to file
  /// @param[in] mesh
  void write_mesh(const mesh::Mesh& mesh);

  /// Write meshtags to file
  /// @param[in] meshtags
  void write_meshtags(const mesh::MeshTags<std::int32_t>& meshtag);

private:
  /// Templated writer for functions using CG-1 interpolation
  template <typename Scalar>
  void _write_function_at_nodes(
      const std::vector<std::reference_wrapper<const fem::Function<Scalar>>>& u,
      double t);

  /// Templated writer for a single Lagrange function
  template <typename Scalar>
  void _write_lagrange_function(
      std::reference_wrapper<const fem::Function<Scalar>> u, double t);

  // Function for updating vtk schema
  std::set<std::string> update_vtk_point_data();

  /// Wrapper for creating VTKSchema for given input
  std::string VTKSchema(std::set<std::string> point_data,
                        std::set<std::string> cell_data);

  std::shared_ptr<adios2::ADIOS> _adios;
  std::shared_ptr<adios2::IO> _io;
  std::shared_ptr<adios2::Engine> _engine;
  bool _time_dep = false;
  std::string _vtk_scheme;
  std::string _mode;
};

} // namespace io
} // namespace dolfinx

#endif