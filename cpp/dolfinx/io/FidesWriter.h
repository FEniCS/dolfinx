// Copyright (C) 2021 Jørgen S. Dokken and Garth Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_ADIOS2

#include "utils.h"
#include <array>
#include <cassert>
#include <complex>
#include <memory>
#include <mpi.h>
#include <string>
#include <vector>
#include <xtensor/xtensor.hpp>

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
}

namespace dolfinx::io
{

class Adios2Writer
{
protected:
  /// Create an ADIOS2-based writer
  /// @param[in] comm The MPI communicator
  /// @param[in] filename Name of output file
  /// @param[in] tag The ADIOS2 object name
  Adios2Writer(MPI_Comm comm, const std::string& filename,
               const std::string& tag);

  /// Create an ADIOS2-based writer for a mesh
  /// @param[in] comm The MPI communicator
  /// @param[in] filename Name of output file
  /// @param[in] tag The ADIOS2 object name
  /// @param[in] mesh The mesh
  Adios2Writer(MPI_Comm comm, const std::string& filename,
               const std::string& tag, std::shared_ptr<const mesh::Mesh> mesh);

  /// Create an ADIOS2-based writer for a list of complex functions
  /// @param[in] comm The MPI communicator
  /// @param[in] filename Name of output file
  /// @param[in] tag The ADIOS2 object name
  /// @param[in] u List of functions
  Adios2Writer(
      MPI_Comm comm, const std::string& filename, const std::string& tag,
      const std::vector<
          std::shared_ptr<const fem::Function<std::complex<double>>>>& u);

  /// Create an ADIOS2-based writer for a list of complex functions
  /// @param[in] comm The MPI communicator
  /// @param[in] filename Name of output file
  /// @param[in] tag The ADIOS2 object name
  /// @param[in] u List of functions
  Adios2Writer(
      MPI_Comm comm, const std::string& filename, const std::string& tag,
      const std::vector<std::shared_ptr<const fem::Function<double>>>& u);

  /// Destructor
  ~Adios2Writer();

  /// Move constructor
  Adios2Writer(Adios2Writer&& writer) = default;

public:
  /// Close the file
  void close();

protected:
  std::unique_ptr<adios2::ADIOS> _adios;
  std::unique_ptr<adios2::IO> _io;
  std::unique_ptr<adios2::Engine> _engine;

  std::shared_ptr<const mesh::Mesh> _mesh;
  std::vector<std::shared_ptr<const fem::Function<double>>> _functions;
  std::vector<std::shared_ptr<const fem::Function<std::complex<double>>>>
      _complex_functions;
};

/// Extract the cell topology (connectivity) in VTK ordering for all
/// cells the mesh. The VTK 'topology' includes higher-order 'nodes'.
/// The index of a 'node' corresponds to the DOLFINx geometry 'nodes'.
/// @param [in] mesh The mesh
/// @return The cell topology in VTK ordering and in term of the DOLFINx
/// geometry 'nodes'
xt::xtensor<std::int64_t, 2>
extract_vtk_connectivity(std::shared_ptr<const mesh::Mesh> mesh);

/// Output of meshes and functions compatible with the FIDES Paraview
/// reader, see
/// https://fides.readthedocs.io/en/latest/paraview/paraview.html
class FidesWriter : public Adios2Writer
{
public:
  /// Create Fides writer for a mesh
  /// @param[in] comm The MPI communciator
  /// @param[in] filename Name of output file
  /// @param[in] mesh The mesh
  FidesWriter(MPI_Comm comm, const std::string& filename,
              std::shared_ptr<const mesh::Mesh> mesh);

  /// Create Fides writer for list of functions (real)
  /// @param[in] comm The MPI communciator
  /// @param[in] filename Name of output file
  /// @param[in] u List of functions
  FidesWriter(
      MPI_Comm comm, const std::string& filename,
      const std::vector<std::shared_ptr<const fem::Function<double>>>& u);

  /// Create Fides writer for list of functions (complex)
  /// @param[in] comm The MPI communciator
  /// @param[in] filename Name of output file
  /// @param[in] u List of functions
  FidesWriter(
      MPI_Comm comm, const std::string& filename,
      const std::vector<
          std::shared_ptr<const fem::Function<std::complex<double>>>>& u);

  /// Move constructor
  FidesWriter(FidesWriter&& file) = default;

  /// Destructor
  ~FidesWriter() = default;

  /// Write the data in the writer to file for a given time step
  /// @param[in] t The time step
  void write(double t);
};

} // namespace dolfinx::io

#endif
