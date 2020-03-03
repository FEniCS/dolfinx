// Copyright (C) 2012-2016 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "xdmf_mesh.h"
// #include "HDF5File.h"
#include "pugixml.hpp"
#include "xdmf_read.h"
#include "xdmf_utils.h"
// #include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
// #include <boost/lexical_cast.hpp>
// #include <dolfinx/common/MPI.h>
// #include <dolfinx/common/log.h>
// #include <dolfinx/common/utils.h>
// #include <dolfinx/fem/DofMap.h>
// #include <dolfinx/function/Function.h>
// #include <dolfinx/function/FunctionSpace.h>
// #include <dolfinx/mesh/DistributedMeshTools.h>
// #include <dolfinx/mesh/Mesh.h>
// #include <dolfinx/mesh/MeshIterator.h>
// #include <dolfinx/mesh/Topology.h>

using namespace dolfinx;
using namespace dolfinx::io;

//----------------------------------------------------------------------------
std::tuple<
    mesh::CellType,
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
    Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
xdmf_mesh::read_mesh_data(MPI_Comm comm, std::string filename)
{
  // Extract parent filepath (required by HDF5 when XDMF stores relative
  // path of the HDF5 files(s) and the XDMF is not opened from its own
  // directory)
  boost::filesystem::path xdmf_filename(filename);
  const boost::filesystem::path parent_path = xdmf_filename.parent_path();

  if (!boost::filesystem::exists(xdmf_filename))
    throw std::runtime_error("Cannot open XDMF file. File does not exists.");

  // Load XML doc from file
  pugi::xml_document xml_doc;
  pugi::xml_parse_result result = xml_doc.load_file(filename.c_str());
  assert(result);

  // Get XDMF node
  pugi::xml_node xdmf_node = xml_doc.child("Xdmf");
  assert(xdmf_node);

  // Get domain node
  pugi::xml_node domain_node = xdmf_node.child("Domain");
  assert(domain_node);

  // Get grid node
  pugi::xml_node grid_node = domain_node.child("Grid");
  assert(grid_node);

  // Get topology node
  pugi::xml_node topology_node = grid_node.child("Topology");
  assert(topology_node);

  // Get cell type
  const auto cell_type_str = xdmf_utils::get_cell_type(topology_node);

  // Get toplogical dimensions
  mesh::CellType cell_type = mesh::to_type(cell_type_str.first);

  // Get geometry node
  pugi::xml_node geometry_node = grid_node.child("Geometry");
  assert(geometry_node);

  // Determine geometric dimension
  pugi::xml_attribute geometry_type_attr
      = geometry_node.attribute("GeometryType");
  assert(geometry_type_attr);
  int gdim = -1;
  const std::string geometry_type = geometry_type_attr.value();
  if (geometry_type == "XY")
    gdim = 2;
  else if (geometry_type == "XYZ")
    gdim = 3;
  else
  {
    throw std::runtime_error(
        "Cannot determine geometric dimension. GeometryType \"" + geometry_type
        + "\" in XDMF file is unknown or unsupported");
  }

  // Get number of points from Geometry dataitem node
  pugi::xml_node geometry_data_node = geometry_node.child("DataItem");
  assert(geometry_data_node);
  const std::vector<std::int64_t> gdims
      = xdmf_utils::get_dataset_shape(geometry_data_node);
  assert(gdims.size() == 2);
  assert(gdims[1] == gdim);

  // Read geometry data
  const std::vector<double> geometry_data
      = xdmf_read::get_dataset<double>(comm, geometry_data_node, parent_path);
  const std::size_t num_local_points = geometry_data.size() / gdim;
  Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>>
      points(geometry_data.data(), num_local_points, gdim);

  // Get topology dataset node
  pugi::xml_node topology_data_node = topology_node.child("DataItem");
  assert(topology_data_node);
  const std::vector<std::int64_t> tdims
      = xdmf_utils::get_dataset_shape(topology_data_node);
  const int npoint_per_cell = tdims[1];

  // Read topology data
  const std::vector<std::int64_t> topology_data
      = xdmf_read::get_dataset<std::int64_t>(comm, topology_data_node,
                                             parent_path);
  const int num_local_cells = topology_data.size() / npoint_per_cell;
  Eigen::Map<const Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>>
      cells(topology_data.data(), num_local_cells, npoint_per_cell);

  return {cell_type, std::move(points), std::move(cells)};
}
//----------------------------------------------------------------------------
