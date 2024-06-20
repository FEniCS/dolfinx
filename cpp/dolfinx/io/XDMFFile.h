// Copyright (C) 2012-2020 Chris N. Richardson, Garth N. Wells and Michal Habera
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "HDF5Interface.h"
#include <concepts>
#include <dolfinx/common/MPI.h>
#include <dolfinx/mesh/cell_types.h>
#include <filesystem>
#include <memory>
#include <string>
#include <variant>

namespace pugi
{
class xml_node;
class xml_document;
} // namespace pugi

namespace dolfinx::fem
{
template <std::floating_point T>
class CoordinateElement;
template <dolfinx::scalar T, std::floating_point U>
class Function;
} // namespace dolfinx::fem

namespace dolfinx::mesh
{
template <std::floating_point T>
class Geometry;
enum class GhostMode : int;
template <std::floating_point T>
class Mesh;
template <typename T>
class MeshTags;
} // namespace dolfinx::mesh

namespace dolfinx::io
{

/// Read and write mesh::Mesh, fem::Function and other objects in
/// XDMF.
///
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

  /// Move constructor
  XDMFFile(XDMFFile&&) = default;

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
  template <std::floating_point U>
  void write_mesh(const mesh::Mesh<U>& mesh,
                  std::string xpath = "/Xdmf/Domain");

  /// Save Geometry
  /// @param[in] geometry
  /// @param[in] name
  /// @param[in] xpath XPath of a node where Geometry will be inserted
  void write_geometry(const mesh::Geometry<double>& geometry, std::string name,
                      std::string xpath = "/Xdmf/Domain");

  /// Read in Mesh
  /// @param[in] element Element that describes the geometry of a cell
  /// @param[in] mode The type of ghosting/halo to use for the mesh when
  ///   distributed in parallel
  /// @param[in] name
  /// @param[in] xpath XPath where Mesh Grid is located
  /// @return A Mesh distributed on the same communicator as the
  ///   XDMFFile
  mesh::Mesh<double> read_mesh(const fem::CoordinateElement<double>& element,
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
  std::pair<std::variant<std::vector<float>, std::vector<double>>,
            std::array<std::size_t, 2>>
  read_geometry_data(std::string name,
                     std::string xpath = "/Xdmf/Domain") const;

  /// Read information about cell type
  /// @param[in] grid_name Name of Grid for which cell type is needed
  /// @param[in] xpath XPath where Grid is stored
  std::pair<mesh::CellType, int>
  read_cell_type(std::string grid_name, std::string xpath = "/Xdmf/Domain");

  /// @brief Write a fem::Function to file.
  ///
  /// @pre The fem::Function `u` must be (i) a lowest-order (P0)
  /// discontinuous Lagrange element or (ii) a continuous Lagrange
  /// element where the element 'nodes' are the same as the nodes of its
  /// mesh::Mesh. Otherwise an exception is raised.
  ///
  /// @note User interpolation to a suitable Lagrange space may be
  /// required to satisfy the precondition on `u`. The VTX output
  /// (io::VTXWriter) format is recommended over XDMF for discontinuous
  /// and/or high-order spaces.
  ///
  /// @param[in] u Function to write to file.
  /// @param[in] t Time stamp to associate with `u`.
  /// @param[in] mesh_xpath XPath for a Grid under which `u` will be
  /// inserted.
  template <dolfinx::scalar T, std::floating_point U = scalar_value_type_t<T>>
  void write_function(const fem::Function<T, U>& u, double t,
                      std::string mesh_xpath
                      = "/Xdmf/Domain/Grid[@GridType='Uniform'][1]");

  /// Write MeshTags
  /// @param[in] meshtags
  /// @param[in] x Mesh geometry
  /// @param[in] geometry_xpath XPath where Geometry is already stored
  /// in file
  /// @param[in] xpath XPath where MeshTags Grid will be inserted
  template <std::floating_point T>
  void write_meshtags(const mesh::MeshTags<std::int32_t>& meshtags,
                      const mesh::Geometry<T>& x, std::string geometry_xpath,
                      std::string xpath = "/Xdmf/Domain");

  /// Read MeshTags by name
  /// @param[in] mesh The Mesh that the data is defined on
  /// @param[in] name Name of the grid node in the xml file. E.g. "Material" in
  ///                 <Grid Name="Material" GridType="Uniform">
  /// @param[in] attribute_label The name of the attribute to read
  /// @param[in] xpath XPath where MeshTags Grid is stored in file
  mesh::MeshTags<std::int32_t>
  read_meshtags_by_label(const mesh::Mesh<double>& mesh, std::string name,
                        std::string attribute_label,
                        std::string xpath = "/Xdmf/Domain");

  /// Read MeshTags
  /// @param[in] mesh The Mesh that the data is defined on
  /// @param[in] name Name of the grid node in the xml file. E.g. "Material" in
  ///                 <Grid Name="Material" GridType="Uniform">
  /// @param[in] xpath XPath where MeshTags Grid is stored in file
  mesh::MeshTags<std::int32_t>
  read_meshtags(const mesh::Mesh<double>& mesh, std::string name,
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
