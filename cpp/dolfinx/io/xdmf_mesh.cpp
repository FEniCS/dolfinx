// Copyright (C) 2012-2016 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "xdmf_mesh.h"
#include "cells.h"
#include "xdmf_read.h"
#include "xdmf_utils.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <pugixml.hpp>
#include <vector>

using namespace dolfinx;
using namespace dolfinx::io;

//-----------------------------------------------------------------------------
void xdmf_mesh::add_topology_data(MPI_Comm comm, pugi::xml_node& xml_node,
                                  const hid_t h5_id,
                                  const std::string path_prefix,
                                  const mesh::Topology& topology,
                                  const mesh::Geometry& geometry, int dim,
                                  std::span<const std::int32_t> entities)
{
  LOG(INFO) << "Adding topology data to node \"" << xml_node.path('/') << "\"";

  const int tdim = topology.dim();

  if (tdim == 2 and topology.cell_type() == mesh::CellType::prism)
    throw std::runtime_error("More work needed for prism cell");

  // Get entity 'cell' type
  const mesh::CellType entity_cell_type
      = mesh::cell_entity_type(topology.cell_type(), dim, 0);

  const fem::ElementDofLayout cmap_dof_layout
      = geometry.cmap().create_dof_layout();

  // Get number of nodes per entity
  const int num_nodes_per_entity = cmap_dof_layout.num_entity_closure_dofs(dim);

  // FIXME: sort out degree/cell type
  // Get VTK string for cell type
  const std::string vtk_cell_str
      = xdmf_utils::vtk_cell_type_str(entity_cell_type, num_nodes_per_entity);

  pugi::xml_node topology_node = xml_node.append_child("Topology");
  assert(topology_node);
  topology_node.append_attribute("TopologyType") = vtk_cell_str.c_str();

  // Pack topology data
  std::vector<std::int64_t> topology_data;

  const graph::AdjacencyList<std::int32_t>& cells_g = geometry.dofmap();
  auto map_g = geometry.index_map();
  assert(map_g);
  const std::int64_t offset_g = map_g->local_range()[0];

  const std::vector<std::int64_t>& ghosts = map_g->ghosts();
  const std::vector vtk_map = io::cells::transpose(
      io::cells::perm_vtk(entity_cell_type, num_nodes_per_entity));
  auto map_e = topology.index_map(dim);
  assert(map_e);
  if (dim == tdim)
  {
    for (std::int32_t c : entities)
    {
      assert(c < cells_g.num_nodes());
      auto nodes = cells_g.links(c);
      for (std::size_t i = 0; i < nodes.size(); ++i)
      {
        std::int64_t global_index = nodes[vtk_map[i]];
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
    auto e_to_c = topology.connectivity(dim, tdim);
    if (!e_to_c)
      throw std::runtime_error("Mesh is missing entity-cell connectivity.");
    auto c_to_e = topology.connectivity(tdim, dim);
    if (!c_to_e)
      throw std::runtime_error("Mesh is missing cell-entity connectivity.");

    // Tabulate geometry dofs for local entities
    std::vector<std::vector<int>> entity_dofs;
    for (int e = 0; e < mesh::cell_num_entities(topology.cell_type(), dim); ++e)
      entity_dofs.push_back(cmap_dof_layout.entity_closure_dofs(dim, e));

    for (std::int32_t e : entities)
    {
      // Get first attached cell
      std::int32_t c = e_to_c->links(e)[0];

      // Find local number of entity wrt. cell
      auto cell_entities = c_to_e->links(c);
      auto it0 = std::find(cell_entities.begin(), cell_entities.end(), e);
      assert(it0 != cell_entities.end());
      const int local_cell_entity = std::distance(cell_entities.begin(), it0);

      // Get geometry dofs for the entity
      const std::vector<int>& entity_dofs_e = entity_dofs[local_cell_entity];

      auto nodes = cells_g.links(c);
      for (std::size_t i = 0; i < entity_dofs_e.size(); ++i)
      {
        std::int64_t global_index = nodes[entity_dofs_e[vtk_map[i]]];
        if (global_index < map_g->size_local())
          global_index += offset_g;
        else
          global_index = ghosts[global_index - map_g->size_local()];

        topology_data.push_back(global_index);
      }
    }
  }

  assert(topology_data.size() % num_nodes_per_entity == 0);
  const std::int64_t num_entities_local
      = topology_data.size() / num_nodes_per_entity;
  std::int64_t num_entities_global = 0;
  MPI_Allreduce(&num_entities_local, &num_entities_global, 1, MPI_INT64_T,
                MPI_SUM, comm);
  topology_node.append_attribute("NumberOfElements")
      = std::to_string(num_entities_global).c_str();
  topology_node.append_attribute("NodesPerElement") = num_nodes_per_entity;

  // Add topology DataItem node
  const std::string h5_path = path_prefix + std::string("/topology");
  const std::vector<std::int64_t> shape
      = {num_entities_global, num_nodes_per_entity};
  const std::string number_type = "Int";

  const std::int64_t num_local = num_entities_local;
  std::int64_t offset = 0;
  MPI_Exscan(&num_local, &offset, 1, MPI_INT64_T, MPI_SUM, comm);
  const bool use_mpi_io = (dolfinx::MPI::size(comm) > 1);
  xdmf_utils::add_data_item(topology_node, h5_id, h5_path, topology_data,
                            offset, shape, number_type, use_mpi_io);
}
//-----------------------------------------------------------------------------
void xdmf_mesh::add_geometry_data(MPI_Comm comm, pugi::xml_node& xml_node,
                                  const hid_t h5_id,
                                  const std::string path_prefix,
                                  const mesh::Geometry& geometry)
{
  LOG(INFO) << "Adding geometry data to node \"" << xml_node.path('/') << "\"";
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
  const int width = (gdim == 1) ? 2 : gdim;

  std::span<const double> _x = geometry.x();

  int num_values = num_points_local * width;
  std::vector<double> x(num_values, 0.0);

  if (width == 3)
    std::copy_n(_x.data(), num_values, x.begin());
  else
  {
    for (int i = 0; i < num_points_local; ++i)
    {
      std::copy_n(std::next(_x.begin(), 3 * i), gdim,
                  std::next(x.begin(), width * i));
    }
  }

  // Add geometry DataItem node
  const std::string h5_path = path_prefix + std::string("/geometry");
  const std::vector<std::int64_t> shape = {num_points, width};

  const std::int64_t num_local = num_points_local;
  std::int64_t offset = 0;
  MPI_Exscan(&num_local, &offset, 1, MPI_INT64_T, MPI_SUM, comm);
  const bool use_mpi_io = (dolfinx::MPI::size(comm) > 1);
  xdmf_utils::add_data_item(geometry_node, h5_id, h5_path, x, offset, shape, "",
                            use_mpi_io);
}
//----------------------------------------------------------------------------
void xdmf_mesh::add_mesh(MPI_Comm comm, pugi::xml_node& xml_node,
                         const hid_t h5_id, const mesh::Mesh& mesh,
                         const std::string name)
{
  LOG(INFO) << "Adding mesh to node \"" << xml_node.path('/') << "\"";

  // Add grid node and attributes
  pugi::xml_node grid_node = xml_node.append_child("Grid");
  assert(grid_node);
  grid_node.append_attribute("Name") = name.c_str();
  grid_node.append_attribute("GridType") = "Uniform";

  // Add topology node and attributes (including writing data)
  const std::string path_prefix = "/Mesh/" + name;
  const int tdim = mesh.topology().dim();

  // Prepare an array of active cells
  // Writing whole mesh so each cell is active, excl. ghosts
  auto map = mesh.topology().index_map(tdim);
  assert(map);
  const int num_cells = map->size_local();
  std::vector<std::int32_t> cells(num_cells);
  std::iota(cells.begin(), cells.end(), 0);

  add_topology_data(comm, grid_node, h5_id, path_prefix, mesh.topology(),
                    mesh.geometry(), tdim,
                    std::span<std::int32_t>(cells.data(), num_cells));

  // Add geometry node and attributes (including writing data)
  add_geometry_data(comm, grid_node, h5_id, path_prefix, mesh.geometry());
}
//----------------------------------------------------------------------------
std::pair<std::vector<double>, std::array<std::size_t, 2>>
xdmf_mesh::read_geometry_data(MPI_Comm comm, const hid_t h5_id,
                              const pugi::xml_node& node)
{
  // Get geometry node
  pugi::xml_node geometry_node = node.child("Geometry");
  assert(geometry_node);

  // Determine geometric dimension
  pugi::xml_attribute geometry_type_attr
      = geometry_node.attribute("GeometryType");
  assert(geometry_type_attr);
  std::size_t gdim = 0;
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
  const std::vector gdims = xdmf_utils::get_dataset_shape(geometry_data_node);
  assert(gdims.size() == 2);
  assert(gdims[1] == (int)gdim);

  // Read geometry data
  std::vector geometry_data
      = xdmf_read::get_dataset<double>(comm, geometry_data_node, h5_id);
  const std::size_t num_local_nodes = geometry_data.size() / gdim;
  std::array<std::size_t, 2> shape = {num_local_nodes, gdim};

  return {std::move(geometry_data), shape};
}
//----------------------------------------------------------------------------
std::pair<std::vector<std::int64_t>, std::array<std::size_t, 2>>
xdmf_mesh::read_topology_data(MPI_Comm comm, const hid_t h5_id,
                              const pugi::xml_node& node)
{
  // Get topology node
  pugi::xml_node topology_node = node.child("Topology");
  assert(topology_node);

  // Get cell type
  const std::pair<std::string, int> cell_type_str
      = xdmf_utils::get_cell_type(topology_node);

  // Get toplogical dimensions
  mesh::CellType cell_type = mesh::to_type(cell_type_str.first);

  // Get topology dataset node
  pugi::xml_node topology_data_node = topology_node.child("DataItem");
  assert(topology_data_node);
  const std::vector tdims = xdmf_utils::get_dataset_shape(topology_data_node);
  const std::size_t npoint_per_cell = tdims[1];

  // Read topology data
  std::vector<std::int64_t> topology_data
      = xdmf_read::get_dataset<std::int64_t>(comm, topology_data_node, h5_id);
  const std::size_t num_local_cells = topology_data.size() / npoint_per_cell;

  //  Permute cells from VTK to DOLFINx ordering
  std::array<std::size_t, 2> shape = {num_local_cells, npoint_per_cell};
  std::vector<std::int64_t> cells = io::cells::apply_permutation(
      topology_data, shape, io::cells::perm_vtk(cell_type, shape[1]));

  return {std::move(cells), shape};
}
//----------------------------------------------------------------------------
