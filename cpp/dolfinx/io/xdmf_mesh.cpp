// Copyright (C) 2012-2016 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "xdmf_mesh.h"
#include "cells.h"
#include "pugixml.hpp"
#include "xdmf_read.h"
#include "xdmf_utils.h"
#include <boost/filesystem.hpp>
#include <dolfinx/fem/ElementDofLayout.h>

using namespace dolfinx;
using namespace dolfinx::io;

//-----------------------------------------------------------------------------
void xdmf_mesh::add_topology_data(MPI_Comm comm, pugi::xml_node& xml_node,
                                  hid_t& h5_id, const std::string path_prefix,
                                  const mesh::Topology& topology,
                                  const mesh::Geometry& geometry, int dim)
{
  const int tdim = topology.dim();

  // Get number of cells (global) and vertices per cell from mesh
  auto map_e = topology.index_map(dim);
  assert(map_e);
  const std::int64_t num_entities_global = map_e->size_global();

  // Get entity 'cell' type
  const mesh::CellType entity_cell_type
      = mesh::cell_entity_type(topology.cell_type(), dim);

  // Get number of nodes per cell
  const int num_nodes_per_cell
      = geometry.dof_layout().num_entity_closure_dofs(dim);

  // FIXME: sort out degree/cell type
  // Get VTK string for cell type
  const std::string vtk_cell_str
      = xdmf_utils::vtk_cell_type_str(entity_cell_type, num_nodes_per_cell);

  pugi::xml_node topology_node = xml_node.append_child("Topology");
  assert(topology_node);
  topology_node.append_attribute("NumberOfElements")
      = std::to_string(num_entities_global).c_str();
  topology_node.append_attribute("TopologyType") = vtk_cell_str.c_str();

  // Pack topology data
  std::vector<std::int64_t> topology_data;

  const graph::AdjacencyList<std::int32_t>& cells_g = geometry.dofmap();
  auto map_g = geometry.index_map();
  assert(map_g);
  const std::int64_t offset_g = map_g->local_range()[0];

  const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& ghosts = map_g->ghosts();

  const std::vector<std::uint8_t> perm
      = io::cells::vtk_to_dolfin(entity_cell_type, num_nodes_per_cell);

  if (dim == tdim)
  {
    for (int c = 0; c < map_e->size_local(); ++c)
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
  }
  else
  {
    // FIXME: This will not work for higher-order cells. Need to use
    // ElementDofLayout to loop over all nodes
    auto e_to_v = topology.connectivity(dim, 0);
    assert(e_to_v);

    auto e_to_c = topology.connectivity(dim, tdim);
    assert(e_to_c);

    auto c_to_v = topology.connectivity(tdim, 0);
    assert(c_to_v);
    for (int e = 0; e < map_e->size_local(); ++e)
    {
      // Get first attached cell
      std::int32_t c = e_to_c->links(e)[0];
      auto cell_vertices = c_to_v->links(c);
      auto nodes = cells_g.links(c);

      auto vertices = e_to_v->links(e);
      for (int v = 0; v < vertices.rows(); ++v)
      {
        const int v_index = perm[v];
        const int vertex = vertices[v_index];
        auto it
            = std::find(cell_vertices.data(),
                        cell_vertices.data() + cell_vertices.rows(), vertex);
        assert(it != (cell_vertices.data() + cell_vertices.rows()));
        const int local_cell_vertex = std::distance(cell_vertices.data(), it);

        std::int64_t global_index = nodes[local_cell_vertex];
        if (global_index < map_g->size_local())
          global_index += offset_g;
        else
          global_index = ghosts[global_index - map_g->size_local()];

        topology_data.push_back(global_index);
      }
    }
  }

  topology_node.append_attribute("NodesPerElement") = num_nodes_per_cell;

  // Add topology DataItem node
  const std::string group_name = path_prefix + "/" + "mesh";
  const std::string h5_path = group_name + "/topology";
  const std::vector<std::int64_t> shape
      = {num_entities_global, num_nodes_per_cell};
  const std::string number_type = "Int";

  const std::int64_t offset
      = dolfinx::MPI::global_offset(comm, map_e->size_local(), true);

  const bool use_mpi_io = (dolfinx::MPI::size(comm) > 1);
  xdmf_utils::add_data_item(topology_node, h5_id, h5_path, topology_data,
                            offset, shape, number_type, use_mpi_io);
}
//-----------------------------------------------------------------------------
void xdmf_mesh::add_geometry_data(MPI_Comm comm, pugi::xml_node& xml_node,
                                  hid_t h5_id, const std::string path_prefix,
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

  const std::int64_t offset
      = dolfinx::MPI::global_offset(comm, num_points_local, true);
  const bool use_mpi_io = (dolfinx::MPI::size(comm) > 1);
  xdmf_utils::add_data_item(geometry_node, h5_id, h5_path, x, offset, shape, "",
                            use_mpi_io);
}
//----------------------------------------------------------------------------
void xdmf_mesh::add_mesh(MPI_Comm comm, pugi::xml_node& xml_node, hid_t& h5_id,
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

  //  Permute cells from VTK to DOLFINX ordering
  Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      cells1 = io::cells::permute_ordering(
          cells, io::cells::vtk_to_dolfin(cell_type, cells.cols()));

  return {cell_type, std::move(points), std::move(cells1)};
}
//----------------------------------------------------------------------------
