// Copyright (C) 2012-2016 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "XDMFFile.h"
#include "HDF5File.h"
#include "cells.h"
#include "pugixml.hpp"
#include "xdmf_function.h"
#include "xdmf_mesh.h"
#include "xdmf_mf.h"
#include "xdmf_utils.h"
#include <boost/filesystem.hpp>
#include <dolfinx/common/utils.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/graph/Partitioning.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Partitioning.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/TopologyComputation.h>
#include <dolfinx/mesh/utils.h>

using namespace dolfinx;
using namespace dolfinx::io;

namespace
{
//-----------------------------------------------------------------------------

/// Construct HDF5 filename from XDMF filename
std::string get_hdf5_filename(std::string filename)
{
  boost::filesystem::path p(filename);
  p.replace_extension(".h5");
  if (p.string() == filename)
  {
    throw std::runtime_error("Cannot deduce name of HDF5 file from XDMF "
                             "filename. Filename clash. Check XDMF filename");
  }

  return p.string();
}
//-----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
XDMFFile::XDMFFile(MPI_Comm comm, const std::string filename, Encoding encoding)
    : _mpi_comm(comm), _filename(filename), _xml_doc(new pugi::xml_document),
      _encoding(encoding)
{
  // Open files here?

  // Check that encoding
  if (_encoding == Encoding::ASCII and MPI::size(_mpi_comm.comm()) != 1)
  {
    throw std::runtime_error(
        "Cannot write ASCII XDMF in parallel (use HDF5 encoding).");
  }
}
//-----------------------------------------------------------------------------
XDMFFile::~XDMFFile() { close(); }
//-----------------------------------------------------------------------------
void XDMFFile::close()
{
  // Close the HDF5 file
  _hdf5_file.reset();
}
//-----------------------------------------------------------------------------
void XDMFFile::write(const mesh::Mesh& mesh)
{
  // Open a HDF5 file if using HDF5 encoding (truncate)
  hid_t h5_id = -1;
  std::unique_ptr<HDF5File> h5_file;
  if (_encoding == Encoding::HDF5)
  {
    // Open file
    h5_file = std::make_unique<HDF5File>(mesh.mpi_comm(),
                                         get_hdf5_filename(_filename), "w");
    assert(h5_file);

    // Get file handle
    h5_id = h5_file->h5_id();
  }

  // Reset pugi doc
  _xml_doc->reset();

  // Add XDMF node and version attribute
  _xml_doc->append_child(pugi::node_doctype)
      .set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");
  pugi::xml_node xdmf_node = _xml_doc->append_child("Xdmf");
  assert(xdmf_node);
  xdmf_node.append_attribute("Version") = "3.0";
  xdmf_node.append_attribute("xmlns:xi") = "http://www.w3.org/2001/XInclude";

  // Add domain node and add name attribute
  pugi::xml_node domain_node = xdmf_node.append_child("Domain");
  assert(domain_node);

  // Add the mesh Grid to the domain
  xdmf_mesh::add_mesh(_mpi_comm.comm(), domain_node, h5_id, mesh, "/Mesh");

  // Save XML file (on process 0 only)
  if (MPI::rank(_mpi_comm.comm()) == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");
}
//-----------------------------------------------------------------------------
mesh::Mesh XDMFFile::read_mesh() const
{
  // Read mesh data
  auto [cell_type, x, cells]
      = xdmf_mesh::read_mesh_data(_mpi_comm.comm(), _filename);

  graph::AdjacencyList<std::int64_t> cells_adj(cells);

  // TODO: create outside
  // Create a layout
  const fem::ElementDofLayout layout
      = fem::geometry_layout(cell_type, cells.cols());

  // TODO: This step can be skipped for 'P1' elements
  //
  // Extract topology data, e.g. just the vertices. For P1 geometry this
  // should just be the identity operator. For other elements the
  // filtered lists may have 'gaps', i.e. the indices might not be
  // contiguous.
  const graph::AdjacencyList<std::int64_t> cells_topology
      = mesh::extract_topology(layout, cells_adj);

  // Compute the destination rank for cells on this process via graph
  // partitioning
  const int size = dolfinx::MPI::size(_mpi_comm.comm());
  const graph::AdjacencyList<std::int32_t> dest
      = mesh::Partitioning::partition_cells(_mpi_comm.comm(), size,
                                            layout.cell_type(), cells_topology,
                                            mesh::GhostMode::none);

  // Distribute cells to destination rank
  const auto [cell_nodes, src, original_cell_index, ghost_owners]
      = graph::Partitioning::distribute(_mpi_comm.comm(), cells_adj, dest);

  // Create Topology
  graph::AdjacencyList<std::int64_t> _cells(cells);
  mesh::Topology topology = mesh::create_topology(
      _mpi_comm.comm(), mesh::extract_topology(layout, cell_nodes),
      original_cell_index, ghost_owners, layout, mesh::GhostMode::none);

  // FIXME: Figure out how to check which entities are required
  // Initialise facet for P2
  // Create local entities
  auto [cell_entity, entity_vertex, index_map]
      = mesh::TopologyComputation::compute_entities(_mpi_comm.comm(), topology,
                                                    1);
  if (cell_entity)
    topology.set_connectivity(cell_entity, topology.dim(), 1);
  if (entity_vertex)
    topology.set_connectivity(entity_vertex, 1, 0);
  if (index_map)
    topology.set_index_map(1, index_map);

  // Create Geometry
  const mesh::Geometry geometry = mesh::create_geometry(
      _mpi_comm.comm(), topology, layout, cell_nodes, x);

  // Return Mesh
  return mesh::Mesh(_mpi_comm.comm(), topology, geometry);
}
//-----------------------------------------------------------------------------
std::tuple<
    mesh::CellType,
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
    Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
XDMFFile::read_mesh_data() const
{
  return xdmf_mesh::read_mesh_data(_mpi_comm.comm(), _filename);
}
//-----------------------------------------------------------------------------
void XDMFFile::write(const mesh::MeshFunction<int>& meshfunction)
{
  // Check if _xml_doc already has data. If not, create an outer structure
  // If it already has data, then we may append to it.

  pugi::xml_node domain_node;
  std::string hdf_filemode = "a";
  if (_xml_doc->child("Xdmf").empty())
  {
    // Reset pugi
    _xml_doc->reset();

    // Add XDMF node and version attribute
    _xml_doc->append_child(pugi::node_doctype)
        .set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");
    pugi::xml_node xdmf_node = _xml_doc->append_child("Xdmf");
    assert(xdmf_node);
    xdmf_node.append_attribute("Version") = "3.0";
    xdmf_node.append_attribute("xmlns:xi") = "http://www.w3.org/2001/XInclude";

    // Add domain node and add name attribute
    domain_node = xdmf_node.append_child("Domain");
    hdf_filemode = "w";
  }
  else
    domain_node = _xml_doc->child("Xdmf").child("Domain");

  // Open a HDF5 file if using HDF5 encoding
  hid_t h5_id = -1;
  std::unique_ptr<HDF5File> h5_file;
  if (_encoding == Encoding::HDF5)
  {
    // Open file
    h5_file = std::make_unique<HDF5File>(
        _mpi_comm.comm(), xdmf_utils::get_hdf5_filename(_filename),
        hdf_filemode);
    assert(h5_file);

    // Get file handle
    h5_id = h5_file->h5_id();
  }

  xdmf_mf::write_mesh_function(_mpi_comm.comm(), meshfunction, domain_node,
                               h5_id, _counter);

  // Save XML file (on process 0 only)
  if (MPI::rank(_mpi_comm.comm()) == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");

  // Increment the counter, so we can save multiple mesh::MeshFunctions in one
  // file
  ++_counter;
}
//-----------------------------------------------------------------------------
mesh::MeshFunction<int>
XDMFFile::read_mf_int(std::shared_ptr<const mesh::Mesh> mesh,
                      std::string name) const
{
  // Load XML doc from file
  pugi::xml_document xml_doc;
  pugi::xml_parse_result result = xml_doc.load_file(_filename.c_str());
  assert(result);

  // Get XDMF node
  pugi::xml_node xdmf_node = xml_doc.child("Xdmf");
  assert(xdmf_node);

  // Get domain node
  pugi::xml_node domain_node = xdmf_node.child("Domain");
  assert(domain_node);

  return xdmf_mf::read_mesh_function<int>(mesh, name, _filename, domain_node);
}
//-----------------------------------------------------------------------------
void XDMFFile::write(const function::Function& u, double t)
{
  // Clear the pugi doc the first time
  if (_counter == 0)
  {
    _xml_doc->reset();

    // Create XDMF header
    _xml_doc->append_child(pugi::node_doctype)
        .set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");
    pugi::xml_node xdmf_node = _xml_doc->append_child("Xdmf");
    assert(xdmf_node);
    xdmf_node.append_attribute("Version") = "3.0";
    xdmf_node.append_attribute("xmlns:xi") = "http://www.w3.org/2001/XInclude";
    pugi::xml_node domain_node = xdmf_node.append_child("Domain");
    assert(domain_node);
  }

  hid_t h5_id = -1;
  // Open the HDF5 file for first time, if using HDF5 encoding
  if (_encoding == Encoding::HDF5)
  {
    // Truncate the file the first time
    if (_counter == 0)
      _hdf5_file = std::make_unique<HDF5File>(
          _mpi_comm.comm(), xdmf_utils::get_hdf5_filename(_filename), "w");
    // else if (flush_output)
    // {
    //   // Append to existing HDF5 file
    //   assert(!_hdf5_file);
    //   _hdf5_file = std::make_unique<HDF5File>(
    //       _mpi_comm.comm(), xdmf_utils::get_hdf5_filename(_filename), "a");
    // }
    else if (_counter != 0 and !_hdf5_file)
    {
      // The XDMFFile was previously closed, and now must be reopened
      _hdf5_file = std::make_unique<HDF5File>(
          _mpi_comm.comm(), xdmf_utils::get_hdf5_filename(_filename), "a");
    }
    assert(_hdf5_file);
    h5_id = _hdf5_file->h5_id();
  }

  xdmf_function::write(u, t, _counter, *_xml_doc, h5_id);

  // Save XML file (on process 0 only)
  if (MPI::rank(_mpi_comm.comm()) == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");

  // Close the HDF5 file if in "flush" mode
  // if (_encoding == Encoding::HDF5 and flush_output)
  // {
  //   assert(_hdf5_file);
  //   _hdf5_file.reset();
  // }

  ++_counter;
}
//-----------------------------------------------------------------------------
