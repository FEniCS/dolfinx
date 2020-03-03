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
#include "cells.h"
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
#include <dolfinx/common/utils.h>
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
template <typename T>
void add_data_item(MPI_Comm comm, pugi::xml_node& xml_node, hid_t h5_id,
                   const std::string h5_path, const T& x,
                   const std::vector<std::int64_t> shape,
                   const std::string number_type)
{
  // Add DataItem node
  assert(xml_node);
  pugi::xml_node data_item_node = xml_node.append_child("DataItem");
  assert(data_item_node);

  // Add dimensions attribute
  std::string dims;
  for (auto d : shape)
    dims += std::to_string(d) + " ";
  dims.pop_back();
  data_item_node.append_attribute("Dimensions") = dims.c_str();

  // Set type for topology data (needed by XDMF to prevent default to
  // float)
  if (!number_type.empty())
    data_item_node.append_attribute("NumberType") = number_type.c_str();

  // Add format attribute
  if (h5_id < 0)
  {
    data_item_node.append_attribute("Format") = "XML";
    assert(shape.size() == 2);
    data_item_node.append_child(pugi::node_pcdata)
        .set_value(common::container_to_string(x, " ", 16, shape[1]).c_str());
  }
  else
  {
    data_item_node.append_attribute("Format") = "HDF";

    // Get name of HDF5 file
    const std::string hdf5_filename = HDF5Interface::get_filename(h5_id);
    const boost::filesystem::path p(hdf5_filename);

    // Add HDF5 filename and HDF5 internal path to XML file
    const std::string xdmf_path = p.filename().string() + ":" + h5_path;
    data_item_node.append_child(pugi::node_pcdata).set_value(xdmf_path.c_str());

    // Compute total number of items and check for consistency with shape
    assert(!shape.empty());
    std::int64_t num_items_total = 1;
    for (auto n : shape)
      num_items_total *= n;

    assert(num_items_total == (std::int64_t)dolfinx::MPI::sum(comm, x.size()));

    // Compute data offset and range of values
    std::int64_t local_shape0 = x.size();
    for (std::size_t i = 1; i < shape.size(); ++i)
    {
      assert(local_shape0 % shape[i] == 0);
      local_shape0 /= shape[i];
    }

    // FIXME: Pass in offset
    const std::int64_t offset
        = dolfinx::MPI::global_offset(comm, local_shape0, true);

    const std::array<std::int64_t, 2> local_range
        = {{offset, offset + local_shape0}};

    const bool use_mpi_io = (dolfinx::MPI::size(comm) > 1);
    HDF5Interface::write_dataset(h5_id, h5_path, x.data(), local_range, shape,
                                 use_mpi_io, false);

    // Add partitioning attribute to dataset
    // std::vector<std::size_t> partitions;
    // std::vector<std::size_t> offset_tmp(1, offset);
    // dolfinx::MPI::gather(comm, offset_tmp, partitions);
    // dolfinx::MPI::broadcast(comm, partitions);
    // HDF5Interface::add_attribute(h5_id, h5_path, "partition", partitions);
  }
} // namespace
//----------------------------------------------------------------------------

/// TODO: Document
void add_topology_data(MPI_Comm comm, pugi::xml_node& xml_node, hid_t h5_id,
                       const std::string path_prefix,
                       const mesh::Topology& topology,
                       const mesh::Geometry& geometry, int cell_dim)
{
  const int tdim = topology.dim();
  if (cell_dim != tdim)
  {
    throw std::runtime_error("Cannot create topology data for mesh. "
                             "Can only create mesh of cells");
  }

  // Get number of cells (global) and vertices per cell from mesh
  auto map_c = topology.index_map(cell_dim);
  assert(map_c);
  const std::int64_t num_cells = map_c->size_global();
  int num_nodes_per_cell = mesh::num_cell_vertices(
      mesh::cell_entity_type(topology.cell_type(), cell_dim));

  // FIXME: sort out degree/cell type
  // Get VTK string for cell type
  const std::string vtk_cell_str = xdmf_utils::vtk_cell_type_str(
      mesh::cell_entity_type(topology.cell_type(), cell_dim), 1);

  pugi::xml_node topology_node = xml_node.append_child("Topology");
  assert(topology_node);
  topology_node.append_attribute("NumberOfElements")
      = std::to_string(num_cells).c_str();
  topology_node.append_attribute("TopologyType") = vtk_cell_str.c_str();

  // Pack topology data
  std::vector<std::int64_t> topology_data;

  const graph::AdjacencyList<std::int32_t>& cells_g = geometry.dofmap();
  auto map_g = geometry.index_map();
  assert(map_g);
  const std::int64_t offset_g = map_g->local_range()[0];

  const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& ghosts = map_g->ghosts();

  //   // Adjust num_nodes_per_cell to appropriate size
  //   assert(cells.num_nodes() > 0);
  //   num_nodes_per_cell = cells.num_links(0);
  //   topology_data.reserve(num_nodes_per_cell * cells.num_nodes());

  const std::vector<std::uint8_t> perm
      = io::cells::vtk_to_dolfin(topology.cell_type(), num_nodes_per_cell);
  for (int c = 0; c < map_c->size_local(); ++c)
  {
    assert(c < cells_g.num_nodes());
    auto nodes = cells_g.links(c);
    for (int i = 0; i < nodes.rows(); ++i)
    {
      std::int64_t global_index = nodes[perm[i]];
      if (global_index < map_g->size_local())
        global_index += offset_g;
      else
        global_index = ghosts[global_index - map_g->size_local()];
      topology_data.push_back(global_index);
    }
  }

  topology_node.append_attribute("NodesPerElement") = num_nodes_per_cell;

  // Add topology DataItem node
  const std::string group_name = path_prefix + "/" + "mesh";
  const std::string h5_path = group_name + "/topology";
  const std::vector<std::int64_t> shape = {num_cells, num_nodes_per_cell};
  const std::string number_type = "Int";

  add_data_item(comm, topology_node, h5_id, h5_path, topology_data, shape,
                number_type);
}
//-----------------------------------------------------------------------------

/// TODO: Document
void add_geometry_data(MPI_Comm comm, pugi::xml_node& xml_node, hid_t h5_id,
                       const std::string path_prefix,
                       const mesh::Geometry& geometry)
{
  auto map = geometry.index_map();
  assert(map);

  // Compute number of points (global) in mesh (equal to number of vertices
  // for affine meshes)
  const std::int64_t num_points = map->size_global();
  const std::int32_t num_points_local = map->size_local();

  // Add geometry node and attributes
  int gdim = geometry.dim();
  pugi::xml_node geometry_node = xml_node.append_child("Geometry");
  assert(geometry_node);
  assert(gdim > 0 and gdim <= 3);
  const std::string geometry_type = (gdim == 3) ? "XYZ" : "XY";
  geometry_node.append_attribute("GeometryType") = geometry_type.c_str();

  // Increase 1D to 2D because XDMF has no "X" geometry, use "XY"
  int width = (gdim == 1) ? 2 : gdim;

  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& _x
      = geometry.x();
  int num_values = num_points_local * width;
  std::vector<double> x(num_values, 0.0);
  if (width == 3)
    std::copy(_x.data(), _x.data() + num_values, x.begin());
  else
  {
    for (int i = 0; i < num_points_local; ++i)
    {
      for (int j = 0; j < gdim; ++j)
        x[width * i + j] = _x(i, j);
    }
  }

  // Add geometry DataItem node
  const std::string group_name = path_prefix + "/" + "mesh";
  const std::string h5_path = group_name + "/geometry";
  const std::vector<std::int64_t> shape = {num_points, width};

  add_data_item(comm, geometry_node, h5_id, h5_path, x, shape, "");
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

  // Add topology node and attributes (including writing data)

  const int tdim = mesh.topology().dim();
  add_topology_data(comm, grid_node, h5_id, path_prefix, mesh.topology(),
                    mesh.geometry(), tdim);

  // Add geometry node and attributes (including writing data)
  add_geometry_data(comm, grid_node, h5_id, path_prefix, mesh.geometry());
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
mesh::Mesh XDMFFileNew::read_mesh() const
{

}
//-----------------------------------------------------------------------------
