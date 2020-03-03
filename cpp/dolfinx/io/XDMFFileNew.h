// Copyright (C) 2012-2018 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/MPI.h>
#include <memory>
#include <string>

namespace pugi
{
class xml_node;
class xml_document;
} // namespace pugi

namespace dolfinx
{

namespace mesh
{
class Mesh;
} // namespace mesh

namespace io
{
class HDF5File;

/// Read and write mesh::Mesh, function::Function, mesh::MeshFunction
/// and other objects in XDMF.

/// This class supports the output of meshes and functions in XDMF
/// (http://www.xdmf.org) format. It creates an XML file that describes
/// the data and points to a HDF5 file that stores the actual problem
/// data. Output of data in parallel is supported.
///
/// XDMF is not suitable for checkpointing as it may decimate some data.
/// XDMF is not suitable for higher order geometries, as their currently
/// only supports 1st and 2nd order geometries.

class XDMFFileNew
{
public:
  /// File encoding type
  enum class Encoding
  {
    HDF5,
    ASCII
  };

  /// Default encoding type
  static const Encoding default_encoding = Encoding::HDF5;

  /// Constructor
  XDMFFileNew(MPI_Comm comm, const std::string filename,
              Encoding encoding = default_encoding);

  /// Destructor
  ~XDMFFileNew();

  /// Close the file
  ///
  /// This closes any open HDF5 files. In ASCII mode the XML file is
  /// closed each time it is written to or read from, so close() has
  /// no effect.
  ///
  /// From Python you can also use XDMFFile as a context manager:
  ///
  ///     with XDMFFile(mpi_comm_world(), 'name.xdmf') as xdmf:
  ///         xdmf.write(mesh)
  ///
  /// The file is automatically closed at the end of the with block
  void close();

  /// Save a mesh to XDMF format, either using an associated HDF5 file,
  /// or storing the data inline as XML Create function on given
  /// function space
  /// @param[in] mesh The Mesh to save
  void write(const mesh::Mesh& mesh);

  /// Read in the first Mesh in XDMF file
  /// @return A Mesh distributed on the same communicator as the
  ///   XDMFFile
  mesh::Mesh read_mesh() const;

private:
  // MPI communicator
  dolfinx::MPI::Comm _mpi_comm;

  // HDF5 data file
  std::unique_ptr<HDF5File> _hdf5_file;

  // Cached filename
  const std::string _filename;

  // The XML document currently representing the XDMF which needs to be
  // kept open for time series etc.
  std::unique_ptr<pugi::xml_document> _xml_doc;

  Encoding _encoding;
};

} // namespace io
} // namespace dolfinx
