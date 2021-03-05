// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/MPI.h>
#include <dolfinx/fem/Function.h>
#include <memory>

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
class ADIOSFile
{
public:
  /// Initialize ADIOS
  ADIOSFile(MPI_Comm comm, const std::string filename);

  /// Destructor
  ~ADIOSFile();

  /// Write a function to file
  void write_function(const dolfinx::fem::Function<double>& u);

  void write_function(const dolfinx::fem::Function<std::complex<double>>& u);

  /// Wrapper for creating VTKSchema for given input
  std::string VTKSchema();

private:
  template <typename Scalar>
  void _write_function(const dolfinx::fem::Function<Scalar>& u);

  std::shared_ptr<adios2::ADIOS> _adios;
  // NOTE: Could have separate IOs for different tasks, but we will currently
  // only have one
  std::shared_ptr<adios2::IO> _io;
  std::vector<std::string> _point_data;
  std::shared_ptr<adios2::Engine> _writer;
};

} // namespace io
} // namespace dolfinx