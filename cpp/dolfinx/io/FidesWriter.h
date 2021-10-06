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
#include <variant>
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

class ADIOS2Writer
{
public:
  using U0 = fem::Function<double>;
  using U1 = fem::Function<std::complex<double>>;
  using U = std::vector<
      std::variant<std::shared_ptr<const U0>, std::shared_ptr<const U1>>>;

protected:
  /// Create an ADIOS2-based writer
  /// @param[in] comm The MPI communicator
  /// @param[in] filename Name of output file
  /// @param[in] tag The ADIOS2 object name
  ADIOS2Writer(MPI_Comm comm, const std::string& filename,
               const std::string& tag);

  /// Create an ADIOS2-based writer for a mesh
  /// @param[in] comm The MPI communicator
  /// @param[in] filename Name of output file
  /// @param[in] tag The ADIOS2 object name
  /// @param[in] mesh The mesh
  ADIOS2Writer(MPI_Comm comm, const std::string& filename,
               const std::string& tag, std::shared_ptr<const mesh::Mesh> mesh);

  /// Create an ADIOS2-based writer for a list of functions
  /// @param[in] comm The MPI communicator
  /// @param[in] filename Name of output file
  /// @param[in] tag The ADIOS2 object name
  /// @param[in] u List of functions
  ADIOS2Writer(MPI_Comm comm, const std::string& filename,
               const std::string& tag, const U& u);

  /// Destructor
  ~ADIOS2Writer();

  /// Move constructor
  ADIOS2Writer(ADIOS2Writer&& writer) = default;

public:
  /// Close the file
  void close();

protected:
  std::unique_ptr<adios2::ADIOS> _adios;
  std::unique_ptr<adios2::IO> _io;
  std::unique_ptr<adios2::Engine> _engine;

  std::shared_ptr<const mesh::Mesh> _mesh;
  U _u;

  std::vector<std::variant<std::reference_wrapper<const U0>,
                           std::reference_wrapper<const U1>>>
      Ur;
};

/// Extract the cell topology (connectivity) in VTK ordering for all
/// cells the mesh. The VTK 'topology' includes higher-order 'nodes'.
/// The index of a 'node' corresponds to the DOLFINx geometry 'nodes'.
/// @param [in] mesh The mesh
/// @return The cell topology in VTK ordering and in term of the DOLFINx
/// geometry 'nodes'
xt::xtensor<std::int64_t, 2> extract_vtk_connectivity(const mesh::Mesh& mesh);

/// Output of meshes and functions compatible with the FIDES Paraview
/// reader, see
/// https://fides.readthedocs.io/en/latest/paraview/paraview.html
class FidesWriter : public ADIOS2Writer
{
public:
  /// Create Fides writer for a mesh
  /// @param[in] comm The MPI communicator
  /// @param[in] filename Name of output file
  /// @param[in] mesh The mesh
  FidesWriter(MPI_Comm comm, const std::string& filename,
              std::shared_ptr<const mesh::Mesh> mesh);

  /// Create Fides writer for list of functions (real)
  /// @param[in] comm The MPI communicator
  /// @param[in] filename Name of output file
  /// @param[in] u List of functions
  /// @note All functions in `u` must share the same Mesh
  FidesWriter(MPI_Comm comm, const std::string& filename,
              const ADIOS2Writer::U& u);

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
