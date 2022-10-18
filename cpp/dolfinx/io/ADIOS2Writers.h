// Copyright (C) 2021 Jørgen S. Dokken and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_ADIOS2

#include <cassert>
#include <complex>
#include <filesystem>
#include <memory>
#include <mpi.h>
#include <string>
#include <variant>
#include <vector>

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

/// Base class for ADIOS2-based writers
class ADIOS2Writer
{
public:
  /// Typedefs
  using Fdr = fem::Function<double>;
  /// Typedefs
  using Fdc = fem::Function<std::complex<double>>;
  /// Typedefs
  using U = std::vector<
      std::variant<std::shared_ptr<const Fdr>, std::shared_ptr<const Fdc>>>;

private:
  /// @brief Create an ADIOS2-based writer
  /// @param[in] comm The MPI communicator
  /// @param[in] filename Name of output file
  /// @param[in] tag The ADIOS2 object name
  /// @param[in] mesh
  /// @param[in] u
  ADIOS2Writer(MPI_Comm comm, const std::filesystem::path& filename,
               const std::string& tag, std::shared_ptr<const mesh::Mesh> mesh,
               const U& u);

protected:
  /// @brief Create an ADIOS2-based writer for a mesh
  /// @param[in] comm The MPI communicator
  /// @param[in] filename Name of output file
  /// @param[in] tag The ADIOS2 object name
  /// @param[in] mesh The mesh
  ADIOS2Writer(MPI_Comm comm, const std::filesystem::path& filename,
               const std::string& tag, std::shared_ptr<const mesh::Mesh> mesh);

  /// @brief Create an ADIOS2-based writer for a list of functions
  /// @param[in] comm The MPI communicator
  /// @param[in] filename Name of output file
  /// @param[in] tag The ADIOS2 object name
  /// @param[in] u List of functions
  ADIOS2Writer(MPI_Comm comm, const std::filesystem::path& filename,
               const std::string& tag, const U& u);

  /// @brief Move constructor
  ADIOS2Writer(ADIOS2Writer&& writer) = default;

  /// @brief Copy constructor
  ADIOS2Writer(const ADIOS2Writer&) = delete;

  /// @brief Destructor
  ~ADIOS2Writer();

  /// @brief Move assignment
  ADIOS2Writer& operator=(ADIOS2Writer&& writer) = default;

  // Copy assignment
  ADIOS2Writer& operator=(const ADIOS2Writer&) = delete;

public:
  /// @brief  Close the file
  void close();

protected:
  std::unique_ptr<adios2::ADIOS> _adios;
  std::unique_ptr<adios2::IO> _io;
  std::unique_ptr<adios2::Engine> _engine;
  std::shared_ptr<const mesh::Mesh> _mesh;
  U _u;
};

/// @brief Output of meshes and functions compatible with the Fides Paraview
/// reader, see
/// https://fides.readthedocs.io/en/latest/paraview/paraview.html.
class FidesWriter : public ADIOS2Writer
{
public:
  /// @brief  Create Fides writer for a mesh
  /// @param[in] comm The MPI communicator to open the file on
  /// @param[in] filename Name of output file
  /// @param[in] mesh The mesh. The mesh must a degree 1 mesh.
  /// @note The mesh geometry can be updated between write steps but the
  /// topology should not be changed between write steps
  FidesWriter(MPI_Comm comm, const std::filesystem::path& filename,
              std::shared_ptr<const mesh::Mesh> mesh);

  /// @brief Create Fides writer for list of functions
  /// @param[in] comm The MPI communicator
  /// @param[in] filename Name of output file
  /// @param[in] u List of functions. The functions must (1) share the
  /// same mesh (degree 1) and (2) be degree 1 Lagrange. @note All
  /// functions in `u` must share the same Mesh
  FidesWriter(MPI_Comm comm, const std::filesystem::path& filename,
              const ADIOS2Writer::U& u);

  // Copy constructor
  FidesWriter(const FidesWriter&) = delete;

  /// @brief  Move constructor
  FidesWriter(FidesWriter&& file) = default;

  /// @brief  Destructor
  ~FidesWriter() = default;

  /// @brief Move assignment
  FidesWriter& operator=(FidesWriter&&) = default;

  // Copy assignment
  FidesWriter& operator=(const FidesWriter&) = delete;

  /// @brief Write data with a given time
  /// @param[in] t The time step
  void write(double t);
};

/// @brief Writer for meshes and functions using the ADIOS2 VTX format, see
/// https://adios2.readthedocs.io/en/latest/ecosystem/visualization.html#using-vtk-and-paraview.
///
/// The output files can be visualized using ParaView.
class VTXWriter : public ADIOS2Writer
{
public:
  /// @brief Create a VTX writer for a mesh. This format supports
  /// arbitrary degree meshes.
  /// @param[in] comm The MPI communicator to open the file on
  /// @param[in] filename Name of output file
  /// @param[in] mesh The mesh to write
  /// @note This format support arbitrary degree meshes
  /// @note The mesh geometry can be updated between write steps but the
  /// topology should not be changed between write steps
  VTXWriter(MPI_Comm comm, const std::filesystem::path& filename,
            std::shared_ptr<const mesh::Mesh> mesh);

  /// @brief Create a VTX writer for list of functions
  /// @param[in] comm The MPI communicator to open the file on
  /// @param[in] filename Name of output file
  /// @param[in] u List of functions. The functions must (1) share the
  /// same mesh and (2) be (discontinuous) Lagrange functions. The
  /// element family and degree must be the same for all functions.
  /// @note This format supports arbitrary degree meshes
  VTXWriter(MPI_Comm comm, const std::filesystem::path& filename, const U& u);

  // Copy constructor
  VTXWriter(const VTXWriter&) = delete;

  /// @brief Move constructor
  VTXWriter(VTXWriter&& file) = default;

  /// @brief Destructor
  ~VTXWriter() = default;

  /// @brief Move assignment
  VTXWriter& operator=(VTXWriter&&) = default;

  // Copy assignment
  VTXWriter& operator=(const VTXWriter&) = delete;

  /// @brief  Write data with a given time
  /// @param[in] t The time step
  void write(double t);
};

} // namespace dolfinx::io

#endif
