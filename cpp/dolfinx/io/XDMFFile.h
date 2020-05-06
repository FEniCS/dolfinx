// Copyright (C) 2012-2020 Chris N. Richardson, Garth N. Wells and Michal Habera
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "HDF5Interface.h"
#include "xdmf_meshtags.h"
#include "xdmf_read.h"
#include <dolfinx/common/MPI.h>
#include <dolfinx/mesh/cell_types.h>
#include <memory>
#include <string>

namespace dolfinx
{
namespace fem
{
class CoordinateElement;
}

namespace function
{
class Function;
} // namespace function

namespace mesh
{
class Mesh;
class Geometry;
template <typename T>
class MeshTags;
} // namespace mesh

namespace io
{
class HDF5File;

/// Read and write mesh::Mesh, function::Function and other objects in
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
  XDMFFile(MPI_Comm comm, const std::string filename,
           const std::string file_mode,
           const Encoding encoding = default_encoding);

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
  void write_mesh(const mesh::Mesh& mesh,
                  const std::string xpath = "/Xdmf/Domain");

  /// Save Geometry
  /// @param[in] geometry
  /// @param[in] name
  /// @param[in] xpath XPath of a node where Geometry will be inserted
  void write_geometry(const mesh::Geometry& geometry,
                      const std::string name = "geometry",
                      const std::string xpath = "/Xdmf/Domain");

  /// Read in Mesh
  /// @param[in] element Element that describes the geometry of a cell
  /// @param[in] name
  /// @param[in] xpath XPath where Mesh Grid is located
  /// @return A Mesh distributed on the same communicator as the
  ///   XDMFFile
  mesh::Mesh read_mesh(const fem::CoordinateElement& element,
                       const std::string name,
                       const std::string xpath = "/Xdmf/Domain") const;

  /// Read in the data for Mesh
  /// @param[in] name
  /// @param[in] xpath XPath where Mesh Grid data is located
  /// @return (Cell type, degree), points on each process, and cells
  ///   topology (global node indexing)
  std::tuple<
      std::pair<mesh::CellType, int>,
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
      Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic,
                   Eigen::RowMajor>>
  read_mesh_data(const std::string name = "mesh",
                 const std::string xpath = "/Xdmf/Domain") const;

  /// Write Function
  /// @param[in] function
  /// @param[in] t Time
  /// @param[in] mesh_xpath XPath for a Grid under which Function will
  ///   be inserted
  void write_function(const function::Function& function, const double t,
                      const std::string mesh_xpath
                      = "/Xdmf/Domain/Grid[@GridType='Uniform'][1]");

  /// Write MeshTags
  /// @param[in] meshtags
  /// @param[in] geometry_xpath XPath where Geometry is already stored
  ///   in file
  /// @param[in] xpath XPath where MeshTags Grid will be inserted
  template <typename T>
  void write_meshtags(const mesh::MeshTags<T>& meshtags,
                      const std::string geometry_xpath
                      = "/Xdmf/Domain/Geometry",
                      const std::string xpath = "/Xdmf/Domain");

  /// Read MeshTags
  /// @param[in] mesh
  /// @param[in] name
  /// @param[in] xpath XPath where MeshTags Grid is stored in file
  template <typename T>
  mesh::MeshTags<T>
  read_meshtags(const std::shared_ptr<const mesh::Mesh>& mesh,
                const std::string name,
                const std::string xpath = "/Xdmf/Domain");

  /// Get the MPI communicator
  /// @return The MPI communicator for the file object
  MPI_Comm comm() const;

private:
  // MPI communicator
  dolfinx::MPI::Comm _mpi_comm;

  // Cached filename
  std::string _filename;

  // File mode
  std::string _file_mode;

  // HDF5 file handle
  hid_t _h5_id;

  // The XML document currently representing the XDMF which needs to be
  // kept open for time series etc.
  std::unique_ptr<pugi::xml_document> _xml_doc;

  Encoding _encoding;
};
//---------------------------------------------------------------------------
// Implementation
//---------------------------------------------------------------------------
template <typename T>
void XDMFFile::write_meshtags(const mesh::MeshTags<T>& meshtags,
                              const std::string geometry_xpath,
                              const std::string xpath)
{
  pugi::xml_node node = _xml_doc->select_node(xpath.c_str()).node();
  if (!node)
    throw std::runtime_error("XML node '" + xpath + "' not found.");

  pugi::xml_node grid_node = node.append_child("Grid");
  assert(grid_node);
  grid_node.append_attribute("Name") = meshtags.name.c_str();
  grid_node.append_attribute("GridType") = "Uniform";

  const std::string geo_ref_path = "xpointer(" + geometry_xpath + ")";
  pugi::xml_node geo_ref_node = grid_node.append_child("xi:include");
  geo_ref_node.append_attribute("xpointer") = geo_ref_path.c_str();
  assert(geo_ref_node);
  xdmf_meshtags::add_meshtags(_mpi_comm.comm(), meshtags, grid_node, _h5_id,
                              meshtags.name);

  // Save XML file (on process 0 only)
  if (MPI::rank(_mpi_comm.comm()) == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");
}
//-----------------------------------------------------------------------------
template <typename T>
mesh::MeshTags<T>
XDMFFile::read_meshtags(const std::shared_ptr<const mesh::Mesh>& mesh,
                        const std::string name, const std::string xpath)
{
  pugi::xml_node node = _xml_doc->select_node(xpath.c_str()).node();
  if (!node)
    throw std::runtime_error("XML node '" + xpath + "' not found.");
  pugi::xml_node grid_node
      = node.select_node(("Grid[@Name='" + name + "']").c_str()).node();
  if (!grid_node)
    throw std::runtime_error("<Grid> with name '" + name + "' not found.");

  pugi::xml_node topology_node = grid_node.child("Topology");

  // Get topology dataset node
  pugi::xml_node topology_data_node = topology_node.child("DataItem");
  const std::vector<std::int64_t> tdims
      = xdmf_utils::get_dataset_shape(topology_data_node);

  // Read topology data
  const std::vector<std::int64_t> topology_data
      = xdmf_read::get_dataset<std::int64_t>(_mpi_comm.comm(),
                                             topology_data_node, _h5_id);

  const std::int32_t num_local_entities
      = (std::int32_t)topology_data.size() / tdims[1];
  Eigen::Map<const Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>>
      topology(topology_data.data(), num_local_entities, tdims[1]);

  // Fetch cell type of meshtags and deduce its dimension
  const auto cell_type_str = xdmf_utils::get_cell_type(topology_node);
  const mesh::CellType cell_type = mesh::to_type(cell_type_str.first);
  pugi::xml_node values_data_node
      = grid_node.child("Attribute").child("DataItem");
  std::vector<T> values
      = xdmf_read::get_dataset<T>(_mpi_comm.comm(), values_data_node, _h5_id);
  mesh::MeshTags meshtags = mesh::create_meshtags<T>(
      _mpi_comm.comm(), mesh, cell_type, topology, std::move(values));
  meshtags.name = name;

  return meshtags;
}

} // namespace io
} // namespace dolfinx
