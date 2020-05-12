// Copyright (C) 2012-2020 Chris N. Richardson, Garth N. Wells and Michal Habera
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
#include "xdmf_meshtags.h"
#include "xdmf_read.h"
#include "xdmf_utils.h"
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <dolfinx/common/utils.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/mesh/utils.h>

using namespace dolfinx;
using namespace dolfinx::io;

//-----------------------------------------------------------------------------
XDMFFile::XDMFFile(MPI_Comm comm, const std::string filename,
                   const std::string file_mode, const Encoding encoding)
    : _mpi_comm(comm), _filename(filename), _file_mode(file_mode),
      _xml_doc(new pugi::xml_document), _encoding(encoding)
{
  // Handle HDF5 and XDMF files with the file mode. At the end of this
  // we will have _hdf5_file and _xml_doc both pointing to a valid and
  // opened file handles.

  if (_encoding == Encoding::HDF5)
  {
    // See https://www.hdfgroup.org/hdf5-quest.html#gzero on zero for
    // _hdf5_file_id(0)

    // Open HDF5 file
    const std::string hdf5_filename = xdmf_utils::get_hdf5_filename(_filename);
    const bool mpi_io = MPI::size(_mpi_comm.comm()) > 1 ? true : false;
    _h5_id = HDF5Interface::open_file(_mpi_comm.comm(), hdf5_filename,
                                      file_mode, mpi_io);
    assert(_h5_id > 0);
    LOG(INFO) << "Opened HDF5 file with id \"" << _h5_id << "\"";
  }
  else
  {
    // HDF handle be -1 to avoid closing a HDF file on destruction
    _h5_id = -1;
  }

  if (_file_mode == "r")
  {
    // Load XML doc from file
    pugi::xml_parse_result result = _xml_doc->load_file(_filename.c_str());
    assert(result);

    if (_xml_doc->child("Xdmf").empty())
      throw std::runtime_error("Empty <Xdmf> root node.");

    if (_xml_doc->child("Xdmf").child("Domain").empty())
      throw std::runtime_error("Empty <Domain> node.");
  }
  else if (_file_mode == "w")
  {
    _xml_doc->reset();

    // Add XDMF node and version attribute
    _xml_doc->append_child(pugi::node_doctype)
        .set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");
    pugi::xml_node xdmf_node = _xml_doc->append_child("Xdmf");
    assert(xdmf_node);
    xdmf_node.append_attribute("Version") = "3.0";
    xdmf_node.append_attribute("xmlns:xi") = "http://www.w3.org/2001/XInclude";

    pugi::xml_node domain_node = xdmf_node.append_child("Domain");
    assert(domain_node);
  }
  else if (_file_mode == "a")
  {
    if (boost::filesystem::exists(_filename))
    {
      // Load XML doc from file
      pugi::xml_parse_result result = _xml_doc->load_file(_filename.c_str());
      assert(result);

      if (_xml_doc->child("Xdmf").empty())
        throw std::runtime_error("Empty <Xdmf> root node.");

      if (_xml_doc->child("Xdmf").child("Domain").empty())
        throw std::runtime_error("Empty <Domain> node.");
    }
    else
    {
      _xml_doc->reset();

      // Add XDMF node and version attribute
      _xml_doc->append_child(pugi::node_doctype)
          .set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");
      pugi::xml_node xdmf_node = _xml_doc->append_child("Xdmf");
      assert(xdmf_node);
      xdmf_node.append_attribute("Version") = "3.0";
      xdmf_node.append_attribute("xmlns:xi")
          = "http://www.w3.org/2001/XInclude";

      pugi::xml_node domain_node = xdmf_node.append_child("Domain");
      assert(domain_node);
    }
  }
}
//-----------------------------------------------------------------------------
XDMFFile::~XDMFFile() { close(); }
//-----------------------------------------------------------------------------
void XDMFFile::close()
{
  if (_h5_id > 0)
    HDF5Interface::close_file(_h5_id);
  _h5_id = -1;
}
//-----------------------------------------------------------------------------
void XDMFFile::write_mesh(const mesh::Mesh& mesh, const std::string xpath)
{
  pugi::xml_node node = _xml_doc->select_node(xpath.c_str()).node();
  if (!node)
    throw std::runtime_error("XML node '" + xpath + "' not found.");

  // Add the mesh Grid to the domain
  xdmf_mesh::add_mesh(_mpi_comm.comm(), node, _h5_id, mesh, mesh.name);

  // Save XML file (on process 0 only)
  if (MPI::rank(_mpi_comm.comm()) == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");
}
//-----------------------------------------------------------------------------
void XDMFFile::write_geometry(const mesh::Geometry& geometry,
                              const std::string name, const std::string xpath)
{
  pugi::xml_node node = _xml_doc->select_node(xpath.c_str()).node();
  if (!node)
    throw std::runtime_error("XML node '" + xpath + "' not found.");

  // Prepare a Grid for Geometry only
  pugi::xml_node grid_node = node.append_child("Grid");
  assert(grid_node);
  grid_node.append_attribute("Name") = name.c_str();
  grid_node.append_attribute("GridType") = "Uniform";

  const std::string path_prefix = "/Geometry/" + name;
  xdmf_mesh::add_geometry_data(_mpi_comm.comm(), grid_node, _h5_id, path_prefix,
                               geometry);

  // Save XML file (on process 0 only)
  if (MPI::rank(_mpi_comm.comm()) == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");
}
//-----------------------------------------------------------------------------
mesh::Mesh XDMFFile::read_mesh(const fem::CoordinateElement& element,
                               const mesh::GhostMode& mode,
                               const std::string name,
                               const std::string xpath) const
{
  // Read mesh data
  const auto [cell_type, x, cells] = XDMFFile::read_mesh_data(name, xpath);

  // Create mesh
  graph::AdjacencyList<std::int64_t> cells_adj(cells);
  mesh::Mesh mesh = mesh::create(_mpi_comm.comm(), cells_adj, element, x, mode);
  mesh.name = name;
  return mesh;
}
//-----------------------------------------------------------------------------
std::tuple<
    std::pair<mesh::CellType, int>,
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
    Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
XDMFFile::read_mesh_data(const std::string name, const std::string xpath) const
{
  pugi::xml_node node = _xml_doc->select_node(xpath.c_str()).node();
  if (!node)
    throw std::runtime_error("XML node '" + xpath + "' not found.");

  pugi::xml_node grid_node
      = node.select_node(("Grid[@Name='" + name + "']").c_str()).node();
  if (!grid_node)
    throw std::runtime_error("<Grid> with name '" + name + "' not found.");

  return xdmf_mesh::read_mesh_data(_mpi_comm.comm(), _h5_id, grid_node);
}
//-----------------------------------------------------------------------------
void XDMFFile::write_function(const function::Function& function,
                              const double t, const std::string mesh_xpath)
{
  const std::string timegrid_xpath
      = "/Xdmf/Domain/Grid[@GridType='Collection'][@Name='" + function.name
        + "']";
  pugi::xml_node timegrid_node
      = _xml_doc->select_node(timegrid_xpath.c_str()).node();

  if (!timegrid_node)
  {
    pugi::xml_node domain_node = _xml_doc->select_node("/Xdmf/Domain").node();
    timegrid_node = domain_node.append_child("Grid");
    timegrid_node.append_attribute("Name") = function.name.c_str();
    timegrid_node.append_attribute("GridType") = "Collection";
    timegrid_node.append_attribute("CollectionType") = "Temporal";
  }

  assert(timegrid_node);

  pugi::xml_node grid_node = timegrid_node.append_child("Grid");
  assert(grid_node);
  grid_node.append_attribute("Name") = function.name.c_str();
  grid_node.append_attribute("GridType") = "Uniform";

  pugi::xml_node mesh_node = _xml_doc->select_node(mesh_xpath.c_str()).node();
  if (!mesh_node)
    LOG(WARNING) << "No mesh found at '" << mesh_xpath
                 << "'. Write mesh before function!";

  const std::string ref_path
      = "xpointer(" + mesh_xpath + "/*[self::Topology or self::Geometry])";

  pugi::xml_node topo_geo_ref = grid_node.append_child("xi:include");
  topo_geo_ref.append_attribute("xpointer") = ref_path.c_str();
  assert(topo_geo_ref);

  std::string t_str = boost::lexical_cast<std::string>(t);
  pugi::xml_node time_node = grid_node.append_child("Time");
  time_node.append_attribute("Value") = t_str.c_str();
  assert(time_node);

  // Add the mesh Grid to the domain
  xdmf_function::add_function(_mpi_comm.comm(), function, t, grid_node, _h5_id);

  // Save XML file (on process 0 only)
  if (MPI::rank(_mpi_comm.comm()) == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");
}
//-----------------------------------------------------------------------------
void XDMFFile::write_meshtags(const mesh::MeshTags<std::int32_t>& meshtags,
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
mesh::MeshTags<std::int32_t>
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
  std::vector<std::int32_t> values = xdmf_read::get_dataset<std::int32_t>(
      _mpi_comm.comm(), values_data_node, _h5_id);
  mesh::MeshTags meshtags = mesh::create_meshtags(
      _mpi_comm.comm(), mesh, cell_type, topology, std::move(values));
  meshtags.name = name;

  return meshtags;
}
//-----------------------------------------------------------------------------
MPI_Comm XDMFFile::comm() const { return _mpi_comm.comm(); }
//-----------------------------------------------------------------------------
