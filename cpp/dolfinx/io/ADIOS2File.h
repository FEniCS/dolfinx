// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_ADIOS2

#include <dolfinx/common/MPI.h>
#include <dolfinx/fem/Function.h>
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

  /// Write a function to file
  void write_function(
      const std::vector<std::reference_wrapper<const fem::Function<double>>>& u,
      double t = 0.0);

  void write_function(
      const std::vector<
          std::reference_wrapper<const fem::Function<std::complex<double>>>>& u,
      double t = 0.0);

private:
  template <typename Scalar>
  void _write_function(
      const std::vector<std::reference_wrapper<const fem::Function<Scalar>>>& u,
      double t);
  // Function for updating vtk schema
  std::set<std::string> update_vtk_point_data();

  /// Wrapper for creating VTKSchema for given input
  std::string VTKSchema(std::set<std::string> point_data);

  std::shared_ptr<adios2::ADIOS> _adios;
  // NOTE: Could have separate IOs for different tasks, but we will currently
  // only have one
  std::shared_ptr<adios2::IO> _io;
  std::shared_ptr<adios2::Engine> _engine;
  bool _time_dep = false;
  // String holding vtk scheme from previous time step (in append mode)
  std::string _vtk_scheme;
  std::string _mode;
};

} // namespace io
} // namespace dolfinx

#endif