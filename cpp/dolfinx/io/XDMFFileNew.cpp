// Copyright (C) 2012-2016 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

// #include "xdmf_read.h"
// #include "xdmf_utils.h"
// #include "xdmf_write.h"

#include "HDF5File.h"
// #include "HDF5Utility.h"
#include "XDMFFileNew.h"
// #include "cells.h"
#include "pugixml.hpp"
#include "xdmf_utils.h"
// #include <algorithm>
// #include <boost/algorithm/string.hpp>
// #include <boost/container/vector.hpp>
#include <boost/filesystem.hpp>
// #include <boost/format.hpp>
// #include <boost/lexical_cast.hpp>
// #include <dolfinx/common/MPI.h>
// #include <dolfinx/common/defines.h>
// #include <dolfinx/common/log.h>
// #include <dolfinx/common/utils.h>
// #include <dolfinx/fem/DofMap.h>
// #include <dolfinx/function/Function.h>
// #include <dolfinx/function/FunctionSpace.h>
// #include <dolfinx/graph/AdjacencyList.h>
// #include <dolfinx/la/PETScVector.h>
// #include <dolfinx/la/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
// #include <dolfinx/mesh/MeshEntity.h>
// #include <dolfinx/mesh/MeshIterator.h>
// #include <dolfinx/mesh/MeshValueCollection.h>
// #include <dolfinx/mesh/Partitioning.h>
// #include <iomanip>
// #include <memory>
// #include <petscvec.h>
// #include <set>
// #include <string>
// #include <vector>

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

/// TODO: Document
void add_topology_data(MPI_Comm comm, pugi::xml_node& xml_node, hid_t h5_id,
                       const std::string path_prefix, const mesh::Mesh& mesh,
                       int cell_dim)
{
  // Get number of cells (global) and vertices per cell from mesh
  auto map_c = mesh.topology().index_map(cell_dim);
  assert(map_c);
  const std::int64_t num_cells = map_c->size_global();
  int num_nodes_per_cell = mesh::num_cell_vertices(
      mesh::cell_entity_type(mesh.topology().cell_type(), cell_dim));

  // FIXME: sort out degree/cell type
  // Get VTK string for cell type
  const std::string vtk_cell_str = xdmf_utils::vtk_cell_type_str(
      mesh::cell_entity_type(mesh.topology().cell_type(), cell_dim), 1);

  pugi::xml_node topology_node = xml_node.append_child("Topology");
  assert(topology_node);
  topology_node.append_attribute("NumberOfElements")
      = std::to_string(num_cells).c_str();
  topology_node.append_attribute("TopologyType") = vtk_cell_str.c_str();

  // Pack topology data
  std::vector<std::int64_t> topology_data;

  const int tdim = mesh.topology().dim();
  if (cell_dim != tdim)
  {
    throw std::runtime_error("Cannot create topology data for mesh. "
                             "Can only create mesh of cells");
  }

  //   const auto& global_indices = mesh.geometry().global_indices();
  //   const graph::AdjacencyList<std::int32_t>& cells =
  //   mesh.geometry().dofmap();

  //   // Adjust num_nodes_per_cell to appropriate size
  //   assert(cells.num_nodes() > 0);
  //   num_nodes_per_cell = cells.num_links(0);
  //   topology_data.reserve(num_nodes_per_cell * cells.num_nodes());

  //   const std::vector<std::uint8_t> perm = io::cells::vtk_to_dolfin(
  //       mesh.topology().cell_type(), num_nodes_per_cell);
  //   for (int c = 0; c < cells.num_nodes(); ++c)
  //   {
  //     auto nodes = cells.links(c);
  //     for (int i = 0; i < nodes.rows(); ++i)
  //       topology_data.push_back(global_indices[nodes[perm[i]]]);
  //   }
  // }
  // else
  //   topology_data = compute_topology_data(mesh, cell_dim);

  // topology_node.append_attribute("NodesPerElement") = num_nodes_per_cell;

  // // Add topology DataItem node
  // const std::string group_name = path_prefix + "/" + "mesh";
  // const std::string h5_path = group_name + "/topology";
  // const std::vector<std::int64_t> shape = {num_cells, num_nodes_per_cell};
  // const std::string number_type = "Int";

  // xdmf_write::add_data_item(comm, topology_node, h5_id, h5_path,
  // topology_data,
  //                           shape, number_type);
}
//-----------------------------------------------------------------------------

/// TODO: Document
void add_mesh(MPI_Comm comm, pugi::xml_node& xml_node, hid_t h5_id,
              const mesh::Mesh& mesh, const std::string path_prefix)
{
  LOG(INFO) << "Adding mesh to node \"" << xml_node.path('/') << "\"";

  // Add grid node and attributes
  pugi::xml_node grid_node = xml_node.append_child("Grid");
  assert(grid_node);
  grid_node.append_attribute("Name") = "mesh";
  grid_node.append_attribute("GridType") = "Uniform";

  // // Add topology node and attributes (including writing data)
  const int tdim = mesh.topology().dim();
  add_topology_data(comm, grid_node, h5_id, path_prefix, mesh, tdim);

  // // Add geometry node and attributes (including writing data)
  // add_geometry_data(comm, grid_node, h5_id, path_prefix, mesh.geometry());
}
//----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
XDMFFileNew::XDMFFileNew(MPI_Comm comm, const std::string filename,
                         Encoding encoding)
    : _mpi_comm(comm), _filename(filename), _xml_doc(new pugi::xml_document),
      _encoding(encoding)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XDMFFileNew::~XDMFFileNew() { close(); }
//-----------------------------------------------------------------------------
void XDMFFileNew::close()
{
  // Close the HDF5 file
  _hdf5_file.reset();
}
//-----------------------------------------------------------------------------
void XDMFFileNew::write(const mesh::Mesh& mesh)
{
  // Check that encoding
  if (_encoding == Encoding::ASCII and MPI::size(_mpi_comm.comm()) != 1)
  {
    throw std::runtime_error(
        "Cannot write ASCII XDMF in parallel (use HDF5 encoding).");
  }

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
  add_mesh(_mpi_comm.comm(), domain_node, h5_id, mesh, "/Mesh");

  // Save XML file (on process 0 only)
  if (MPI::rank(_mpi_comm.comm()) == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");
}
//-----------------------------------------------------------------------------
