// Copyright (C) 2012-2018 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "HDF5Interface.h"
#include "pugixml.hpp"
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <dolfin/common/MPI.h>
#include <dolfin/common/utils.h>
#include <dolfin/function/Function.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/MeshIterator.h>
#include <hdf5.h>

// namespace pugi
// {
// class xml_node;
// } // namespace pugi

namespace dolfin
{

namespace function
{
class Function;
}
namespace mesh
{
class Mesh;
}

namespace io
{
namespace xdmf_write
{

// FIXME: do not expose this
/// Calculate set of entities of dimension cell_dim which are duplicated
/// on other processes and should not be output on this process
std::set<std::uint32_t> compute_nonlocal_entities(const mesh::Mesh& mesh,
                                                  int cell_dim);

/// Add set of points to XDMF xml_node and write data
void add_points(MPI_Comm comm, pugi::xml_node& xdmf_node, hid_t h5_id,
                const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 3,
                                                    Eigen::RowMajor>>
                    points);

/// Add set of points to XDMF xml_node and write data
template <typename T>
void add_information(MPI_Comm comm, pugi::xml_node& information_node,
                const std::map<std::string, T>& information){
  information_node.append_attribute("Name") = "Information";
  information_node.append_attribute("Value") = information.size();
  auto it = information.begin();

  pugi::xml_document cdata_doc;
  pugi::xml_node main_node = cdata_doc.append_child("main");

  while (it != information.end()) {
    std::string key_tag = it->first;
    T value_tag = it->second;

    pugi::xml_node map_node = main_node.append_child("map");
    map_node.append_attribute("key")
      = key_tag.c_str();

    std::string value_tag_str = boost::lexical_cast<std::string>(value_tag);
    map_node.append_child(pugi::node_pcdata).set_value(value_tag_str.c_str());
    
    it++;
  }
  // Write complete xml document to string stream
  std::stringstream cdata_ss;
  cdata_doc.save(cdata_ss,"  ");
  information_node.append_child(pugi::node_cdata).set_value(cdata_ss.str().c_str());
}
//----------------------------------------------------------------------------

/// Add topology node to xml_node (includes writing data to XML or HDF5
/// file)
void add_topology_data(MPI_Comm comm, pugi::xml_node& xml_node, hid_t h5_id,
                       const std::string path_prefix, const mesh::Mesh& mesh,
                       int cell_dim);

/// Add geometry node and data to xml_node
void add_geometry_data(MPI_Comm comm, pugi::xml_node& xml_node, hid_t h5_id,
                       const std::string path_prefix, const mesh::Mesh& mesh);

/// Add mesh to XDMF xml_node (usually a Domain or Time Grid) and write
/// data
void add_mesh(MPI_Comm comm, pugi::xml_node& xml_node, hid_t h5_id,
              const mesh::Mesh& mesh, const std::string path_prefix);

/// Add function to a XML node
void add_function(MPI_Comm mpi_comm, pugi::xml_node& xml_node, hid_t h5_id,
                  std::string h5_path, const function::Function& u,
                  std::string function_name, const mesh::Mesh& mesh,
                  const std::string component);

/// Add DataItem node to an XML node. If HDF5 is open (h5_id > 0) the
/// data is written to the HDFF5 file with the path 'h5_path'. Otherwise,
/// data is witten to the XML node and 'h5_path' is ignored
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
  data_item_node.append_attribute("Dimensions")
      = common::container_to_string(shape, " ", 16).c_str();

  // Set type for topology data (needed by XDMF to prevent default to float)
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

    assert(num_items_total == (std::int64_t)dolfin::MPI::sum(comm, x.size()));

    // Compute data offset and range of values
    std::int64_t local_shape0 = x.size();
    for (std::size_t i = 1; i < shape.size(); ++i)
    {
      assert(local_shape0 % shape[i] == 0);
      local_shape0 /= shape[i];
    }
    const std::int64_t offset
        = dolfin::MPI::global_offset(comm, local_shape0, true);
    const std::array<std::int64_t, 2> local_range
        = {{offset, offset + local_shape0}};

    const bool use_mpi_io = (dolfin::MPI::size(comm) > 1);
    HDF5Interface::write_dataset(h5_id, h5_path, x.data(), local_range, shape,
                                 use_mpi_io, false);

    // Add partitioning attribute to dataset
    std::vector<std::size_t> partitions;
    std::vector<std::size_t> offset_tmp(1, offset);
    dolfin::MPI::gather(comm, offset_tmp, partitions);
    dolfin::MPI::broadcast(comm, partitions);
    HDF5Interface::add_attribute(h5_id, h5_path, "partition", partitions);
  }
}

// Return data which is local
template <typename T>
std::vector<T> compute_value_data(const mesh::MeshFunction<T>& meshfunction)
{
  // Create vector to store data
  std::vector<T> value_data;
  value_data.reserve(meshfunction.values().size());

  // Get mesh communicator
  const auto mesh = meshfunction.mesh();
  MPI_Comm comm = mesh->mpi_comm();

  const int tdim = mesh->topology().dim();
  const int cell_dim = meshfunction.dim();

  if (dolfin::MPI::size(comm) == 1 or cell_dim == tdim)
  {
    // FIXME: fail with ghosts?
    value_data.resize(meshfunction.values().size());
    std::copy(meshfunction.values().data(),
              meshfunction.values().data() + meshfunction.values().size(),
              value_data.begin());
  }
  else
  {
    std::set<std::uint32_t> non_local_entities
        = xdmf_write::compute_nonlocal_entities(*mesh, cell_dim);

    // Get reference to mesh function data array
    Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, 1>> mf_values
        = meshfunction.values();

    for (auto& e : mesh::MeshRange<mesh::MeshEntity>(*mesh, cell_dim))
    {
      if (non_local_entities.find(e.index()) == non_local_entities.end())
        value_data.push_back(mf_values[e.index()]);
    }
  }

  return value_data;
}
//----------------------------------------------------------------------------

} // namespace xdmf_write
} // namespace io
} // namespace dolfin
