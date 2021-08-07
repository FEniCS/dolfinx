// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#ifdef HAS_ADIOS2

#include <adios2.h>

#include "utils.h"
#include <array>
#include <cassert>
#include <memory>
#include <mpi.h>
#include <string>
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

class Adios2Writer
{
protected:
  /// Create an ADIOS2-based writer
  /// @param[in] comm The MPI communicator
  /// @param[in] filename Name of output file
  /// @param[in] tag The ADIOS2 obejct name
  Adios2Writer(MPI_Comm comm, const std::string& filename,
               const std::string& tag)
      : _adios(std::make_unique<adios2::ADIOS>(comm)),
        _io(std::make_unique<adios2::IO>(_adios->DeclareIO(tag))),
        _engine(std::make_unique<adios2::Engine>(
            _io->Open(filename, adios2::Mode::Write)))
  {
    _io->SetEngine("BPFile");
  }

  /// Create an ADIOS2-based writer for a mesh
  /// @param[in] comm The MPI communicator
  /// @param[in] filename Name of output file
  /// @param[in] tag The ADIOS2 obejct name
  /// @param[in] mesh The mesh
  Adios2Writer(MPI_Comm comm, const std::string& filename,
               const std::string& tag, std::shared_ptr<const mesh::Mesh> mesh)
      : Adios2Writer(comm, filename, tag)
  {
    _mesh = mesh;
  }

  /// Create an ADIOS2-based writer for a Functions
  /// @param[in] comm The MPI communicator
  /// @param[in] filename Name of output file
  /// @param[in] tag The ADIOS2 obejct name
  /// @param[in] u List of functions
  Adios2Writer(
      MPI_Comm comm, const std::string& filename, const std::string& tag,
      const std::vector<std::shared_ptr<const fem::Function<double>>>& u)
      : Adios2Writer(comm, filename, tag)
  {
    _u = u;
  }

  /// Destructor
  ~Adios2Writer() { close(); }

  /// Move constructor
  Adios2Writer(Adios2Writer&& writer) = default;

public:
  /// Close the file
  void close()
  {
    assert(_engine);
    // This looks a bit odd because ADIOS2 uses `operator bool()` to
    // test if the engine is open
    if (*_engine)
      _engine->Close();
  }

protected:
  std::unique_ptr<adios2::ADIOS> _adios;
  std::unique_ptr<adios2::IO> _io;
  std::unique_ptr<adios2::Engine> _engine;

  std::shared_ptr<const dolfinx::mesh::Mesh> _mesh;
  std::vector<std::shared_ptr<const fem::Function<double>>> _u;
  // std::vector<std::shared_ptr<const fem::Function<std::complex<double>>>>
  //     _complex_functions;
};

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
  /// @param[in] functions List of functions
  FidesWriter(MPI_Comm comm, const std::string& filename,
              const std::vector<std::shared_ptr<const fem::Function<double>>>&
                  functions);

  // /// Create Fides writer for list of functions (complex)
  // /// @param[in] comm The MPI communciator
  // /// @param[in] filename Name of output file
  // /// @param[in] functions List of functions
  // FidesWriter(MPI_Comm comm, const std::string& filename,
  //             const std::vector<
  //                 std::shared_ptr<const
  //                 fem::Function<std::complex<double>>>>& functions);

  /// Move constructor
  FidesWriter(FidesWriter&& file) = default;

  /// Destructor
  ~FidesWriter() = default;

  /// Write the data in the writer to file for a given time step
  /// @param[in] t The time step
  void write(double t);

private:
  // std::unique_ptr<adios2::ADIOS> _adios;
  // std::unique_ptr<adios2::IO> _io;
  // std::unique_ptr<adios2::Engine> _engine;

  // std::shared_ptr<const dolfinx::mesh::Mesh> _mesh;
  // std::vector<std::shared_ptr<const fem::Function<double>>> _functions;
  // std::vector<std::shared_ptr<const fem::Function<std::complex<double>>>>
  //     _complex_functions;
};

} // namespace dolfinx::io

#endif
