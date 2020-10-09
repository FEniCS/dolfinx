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
#include <dolfinx/fem/ElementDofLayout.h>

using namespace dolfinx;
using namespace dolfinx::io;

//-----------------------------------------------------------------------------
void xdmf_mesh::add_topology_data(
    MPI_Comm comm, pugi::xml_node& xml_node, const hid_t h5_id,
    const std::string path_prefix, const mesh::Topology& topology,
    const mesh::Geometry& geometry, const int dim,
    const std::vector<std::int32_t>& active_entities)
{
  LOG(INFO) << "Adding topology data to node \"" << xml_node.path('/') << "\"";

  const int tdim = topology.dim();

  // Get entity 'cell' type
  const mesh::CellType entity_cell_type
      = mesh::cell_entity_type(topology.cell_type(), dim);

  // Get number of nodes per entity
  const int num_nodes_per_entity
      = geometry.cmap().dof_layout().num_entity_closure_dofs(dim);

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

  const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>& ghosts = map_g->ghosts();

  const std::vector vtk_map = io::cells::transpose(
      io::cells::perm_vtk(entity_cell_type, num_nodes_per_entity));
  auto map_e = topology.index_map(dim);
  assert(map_e);
  if (dim == tdim)
  {
    for (std::int32_t c : active_entities)
    {
      assert(c < cells_g.num_nodes());
      auto nodes = cells_g.links(c);
      for (int i = 0; i < nodes.rows(); ++i)
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

    for (std::int32_t e : active_entities)
    {
      // Get first attached cell
      std::int32_t c = e_to_c->links(e)[0];

      // Find local number of entity wrt. cell
      auto cell_entities = c_to_e->links(c);
      const auto* it0 = std::find(
          cell_entities.data(), cell_entities.data() + cell_entities.rows(), e);
      assert(it0 != (cell_entities.data() + cell_entities.rows()));
      const int local_cell_entity = std::distance(cell_entities.data(), it0);

      // Tabulate geometry dofs for the entity
      const Eigen::Array<int, Eigen::Dynamic, 1> entity_dofs
          = geometry.cmap().dof_layout().entity_closure_dofs(dim,
                                                             local_cell_entity);

      auto nodes = cells_g.links(c);
      for (Eigen::Index i = 0; i < entity_dofs.rows(); ++i)
      {
        std::int64_t global_index = nodes[entity_dofs[vtk_map[i]]];
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
  const std::string h5_path = path_prefix + "/topology";
  const std::vector<std::int64_t> shape
      = {num_entities_global, num_nodes_per_entity};
  const std::string number_type = "Int";

  const std::int64_t offset
      = dolfinx::MPI::global_offset(comm, num_entities_local, true);

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
  const std::string h5_path = path_prefix + "/geometry";
  const std::vector<std::int64_t> shape = {num_points, width};

  const std::int64_t offset
      = dolfinx::MPI::global_offset(comm, num_points_local, true);
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
  std::vector<std::int32_t> active_cells(num_cells);
  std::iota(active_cells.begin(), active_cells.end(), 0);

  add_topology_data(comm, grid_node, h5_id, path_prefix, mesh.topology(),
                    mesh.geometry(), tdim, active_cells);

  // Add geometry node and attributes (including writing data)
  add_geometry_data(comm, grid_node, h5_id, path_prefix, mesh.geometry());
}
//----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
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
  const std::vector gdims = xdmf_utils::get_dataset_shape(geometry_data_node);
  assert(gdims.size() == 2);
  assert(gdims[1] == gdim);

  // Read geometry data
  const std::vector geometry_data
      = xdmf_read::get_dataset<double>(comm, geometry_data_node, h5_id);
  const std::size_t num_local_nodes = geometry_data.size() / gdim;
  return Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::RowMajor>>(geometry_data.data(),
                                                         num_local_nodes, gdim);
}
//----------------------------------------------------------------------------
std::pair<
    dolfinx::mesh::CellType,
    Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
xdmf_mesh::read_topology_data(MPI_Comm comm, const hid_t h5_id,
                              const pugi::xml_node& node, std::int32_t tdim)
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

  // Read topology data
  // Extract topologies if we have a mixed topology
  if (tdims.size() == 1)
  {
    // Read in all data on the first processor as it is a 1D array with variable
    // length per cell
    std::array<std::int64_t, 2> range = {{0, 0}};
    const int mpi_rank = dolfinx::MPI::rank(comm);
    if (mpi_rank == 0)
      range[1] = tdims[0];
    const std::vector topology_data = xdmf_read::get_dataset<std::int64_t>(
        comm, topology_data_node, h5_id, range);

    // XDMF topology cell type map to dolfinx cell_type and number of nodes
    // https://gitlab.kitware.com/xdmf/xdmf/blob/master/XdmfTopologyType.cpp
    std::map<std::int32_t, std::pair<dolfinx::mesh::CellType, std::int32_t>>
        xdmf_to_dolfin = {
            {0x1, {dolfinx::mesh::CellType::point, 1}},
            {0x2, {dolfinx::mesh::CellType::interval, 2}},
            {0x4, {dolfinx::mesh::CellType::triangle, 3}},
            {0x5, {dolfinx::mesh::CellType::quadrilateral, 4}},
            {0x6, {dolfinx::mesh::CellType::tetrahedron, 4}},
            {0x9, {dolfinx::mesh::CellType::hexahedron, 8}},
            {0x22, {dolfinx::mesh::CellType::interval, 3}},
            {0x24, {dolfinx::mesh::CellType::triangle, 6}},
            {0x23, {dolfinx::mesh::CellType::quadrilateral, 9}},
            {0x26, {dolfinx::mesh::CellType::tetrahedron, 10}},
            {0x32, {dolfinx::mesh::CellType::hexahedron, 27}},
        };
    std::map<std::pair<dolfinx::mesh::CellType, std::int32_t>,
             std::vector<std::int64_t>>
        cell_topologies;
    std::uint32_t i = 0;
    while (i < topology_data.size())
    {
      std::pair<dolfinx::mesh::CellType, std::int32_t> cell_type
          = xdmf_to_dolfin[topology_data[i++]];
      if (cell_type.first == dolfinx::mesh::CellType::interval)
      {
        // XDMF stores Line segments as arbitrary order, so next data index is
        // the number of nodes Should always be 2.
        const std::int32_t num_nodes = topology_data[i++];
        assert(num_nodes == 2);
      }
      // Check if cell-type is allready added to cell_topologies;
      if (cell_topologies.find(cell_type) != cell_topologies.end())
      {
        cell_topologies[cell_type].insert(
            cell_topologies[cell_type].end(), topology_data.begin() + i,
            topology_data.begin() + i + cell_type.second);
      }
      else
      {
        std::vector<std::int64_t> cell_topology;
        cell_topology.insert(cell_topology.end(), topology_data.begin() + i,
                             topology_data.begin() + i + cell_type.second);
        cell_topologies.insert({cell_type, cell_topology});
      }
      // Jump to next cell
      i += cell_type.second;
    }

    // If input tdim is -1, find cell with highest topological dimension.
    // Otherwise find cell with topological dimension tdim
    std::int32_t cell_tdim = -1;
    std::pair<dolfinx::mesh::CellType, std::int32_t> cell_id;
    for (std::map<std::pair<dolfinx::mesh::CellType, std::int32_t>,
                  std::vector<std::int64_t>>::iterator iter
         = cell_topologies.begin();
         iter != cell_topologies.end(); ++iter)
    {

      std::int32_t td = dolfinx::mesh::cell_dim(iter->first.first);
      // If topology is of correct dimension
      if (tdim == -1)
      {
        if (td == cell_tdim)
          throw std::runtime_error("Mixed topology meshes not supported");
        else if (td > cell_tdim)
        {
          cell_tdim = td;
          cell_id = iter->first;
        }
      }
      else if (td == tdim)
      {
        cell_tdim = tdim;
        cell_id = iter->first;
      }
    }
    if ((cell_tdim == -1) && (mpi_rank == 0))
      throw std::runtime_error("No cell topology found");

    std::int32_t num_local_cells = 0;
    std::int32_t num_nodes_per_cell = 0;
    std::int32_t cell_type_int;
    if (mpi_rank == 0)
    {
      dolfinx::mesh::CellType ct = cell_id.first;
      num_nodes_per_cell = cell_id.second;
      num_local_cells = cell_topologies[cell_id].size() / num_nodes_per_cell;
      cell_type_int = static_cast<std::int32_t>(ct);
    }
    MPI_Bcast(&cell_type_int, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_nodes_per_cell, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);

    Eigen::Map<const Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>>
        cells_vtk(cell_topologies[cell_id].data(), num_local_cells,
                  num_nodes_per_cell);
    dolfinx::mesh::CellType ct_out
        = static_cast<dolfinx::mesh::CellType>(cell_type_int);
    return std::make_pair(
        ct_out,
        io::cells::compute_permutation(
            cells_vtk, io::cells::perm_vtk(ct_out, num_nodes_per_cell)));
  }
  else
  {
    const std::vector topology_data
        = xdmf_read::get_dataset<std::int64_t>(comm, topology_data_node, h5_id);
    const int npoint_per_cell = tdims[1];
    const int num_local_cells = topology_data.size() / npoint_per_cell;
    Eigen::Map<const Eigen::Array<std::int64_t, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>>
        cells_vtk(topology_data.data(), num_local_cells, npoint_per_cell);

    //  Permute cells from VTK to DOLFINX ordering
    return std::make_pair(
        cell_type,
        io::cells::compute_permutation(
            cells_vtk, io::cells::perm_vtk(cell_type, cells_vtk.cols())));
  }
}
//----------------------------------------------------------------------------
