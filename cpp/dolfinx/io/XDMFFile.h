// Copyright (C) 2012-2020 Chris N. Richardson, Garth N. Wells and Michal Habera
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "HDF5Interface.h"
#include <dolfinx/common/MPI.h>
#include <dolfinx/mesh/cell_types.h>
#include <filesystem>
#include <memory>
#include <string>

namespace pugi
{
class xml_node;
class xml_document;
} // namespace pugi

namespace dolfinx::fem
{
class CoordinateElement;
}

namespace dolfinx::fem
{
template <typename T>
class Function;
}

namespace dolfinx::mesh
{
class Geometry;
enum class GhostMode : int;
class Mesh;
template <typename T>
class MeshTags;
} // namespace dolfinx::mesh

namespace dolfinx::io
{

/// Read and write mesh::Mesh, fem::Function and other objects in
/// XDMF.

/// This class supports the output of meshes and functions in XDMF
/// (http://www.xdmf.org) format. It creates an XML file that describes
/// the data and points to a HDF5 file that stores the actual problem
/// data. Output of data in parallel is supported.
///
/// XDMF is not suitable for higher order geometries, as their currently
/// only supports 1st and 2nd order geometries.

class XDMFFile
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
  XDMFFile(MPI_Comm comm, const std::filesystem::path& filename,
           std::string file_mode, Encoding encoding = default_encoding);

  /// Destructor
  ~XDMFFile();

  /// Close the file
  ///
  /// This closes open underlying HDF5 file. In ASCII mode the XML file
  /// is closed each time it is written to or read from, so close() has
  /// no effect.
  void close();

  /// Save Mesh
  /// @param[in] mesh
  /// @param[in] xpath XPath where Mesh Grid will be written
  void write_mesh(const mesh::Mesh& mesh, std::string xpath = "/Xdmf/Domain");

  /// Save Geometry
  /// @param[in] geometry
  /// @param[in] name
  /// @param[in] xpath XPath of a node where Geometry will be inserted
  void write_geometry(const mesh::Geometry& geometry, std::string name,
                      std::string xpath = "/Xdmf/Domain");

  /// Read in Mesh
  /// @param[in] element Element that describes the geometry of a cell
  /// @param[in] mode The type of ghosting/halo to use for the mesh when
  ///   distributed in parallel
  /// @param[in] name
  /// @param[in] xpath XPath where Mesh Grid is located
  /// @return A Mesh distributed on the same communicator as the
  ///   XDMFFile
  mesh::Mesh read_mesh(const fem::CoordinateElement& element,
                       mesh::GhostMode mode, std::string name,
                       std::string xpath = "/Xdmf/Domain") const;

  /// Read Topology data for Mesh
  /// @param[in] name Name of the mesh (Grid)
  /// @param[in] xpath XPath where Mesh Grid data is located
  /// @return (Cell type, degree), and cells topology (global node indexing)
  std::pair<std::vector<std::int64_t>, std::array<std::size_t, 2>>
  read_topology_data(std::string name,
                     std::string xpath = "/Xdmf/Domain") const;

  /// Read Geometry data for Mesh
  /// @param[in] name Name of the mesh (Grid)
  /// @param[in] xpath XPath where Mesh Grid data is located
  /// @return points on each process
  std::pair<std::vector<double>, std::array<std::size_t, 2>>
  read_geometry_data(std::string name,
                     std::string xpath = "/Xdmf/Domain") const;

  /// Read information about cell type
  /// @param[in] grid_name Name of Grid for which cell type is needed
  /// @param[in] xpath XPath where Grid is stored
  std::pair<mesh::CellType, int>
  read_cell_type(std::string grid_name, std::string xpath = "/Xdmf/Domain");

  /// Write Function
  /// @param[in] u The Function to write to file
  /// @param[in] t The time stamp to associate with the Function
  /// @param[in] mesh_xpath XPath for a Grid under which Function will
  /// be inserted
  void write_function(const fem::Function<double>& u, double t,
                      std::string mesh_xpath
                      = "/Xdmf/Domain/Grid[@GridType='Uniform'][1]");

  /// Write Function
  /// @param[in] u The Function to write to file
  /// @param[in] t The time stamp to associate with the Function
  /// @param[in] mesh_xpath XPath for a Grid under which Function will
  /// be inserted
  void write_function(const fem::Function<std::complex<double>>& u, double t,
                      std::string mesh_xpath
                      = "/Xdmf/Domain/Grid[@GridType='Uniform'][1]");

  /// Write MeshTags
  /// @param[in] meshtags
  /// @param[in] geometry_xpath XPath where Geometry is already stored
  ///   in file
  /// @param[in] xpath XPath where MeshTags Grid will be inserted
  void write_meshtags(const mesh::MeshTags<std::int32_t>& meshtags,
                      std::string geometry_xpath,
                      std::string xpath = "/Xdmf/Domain");

  /// Read MeshTags
  /// @param[in] mesh The Mesh that the data is defined on
  /// @param[in] name
  /// @param[in] xpath XPath where MeshTags Grid is stored in file
  mesh::MeshTags<std::int32_t>
  read_meshtags(std::shared_ptr<const mesh::Mesh> mesh, std::string name,
                std::string xpath = "/Xdmf/Domain");

  /// Write Information
  /// @param[in] name
  /// @param[in] value String to store into Information tag
  /// @param[in] xpath XPath where Information will be inserted
  void write_information(std::string name, std::string value,
                         std::string xpath = "/Xdmf/Domain/");

  /// Read Information
  /// @param[in] name
  /// @param[in] xpath XPath where Information is stored in file
  std::string read_information(std::string name,
                               std::string xpath = "/Xdmf/Domain/");

  /// Get the MPI communicator
  /// @return The MPI communicator for the file object
  MPI_Comm comm() const;

private:
  // MPI communicator
  dolfinx::MPI::Comm _comm;

  // Filename
  std::filesystem::path _filename;

  // File mode
  std::string _file_mode;

  // HDF5 file handle
  hid_t _h5_id;

  // The XML document currently representing the XDMF which needs to be
  // kept open for time series etc.
  std::unique_ptr<pugi::xml_document> _xml_doc;

  Encoding _encoding;
};

} // namespace dolfinx::io
