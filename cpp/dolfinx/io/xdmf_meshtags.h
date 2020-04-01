// Copyright (C) 2020 Michal Habera
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "HDF5Interface.h"
#include "pugixml.hpp"
#include "xdmf_mesh.h"
#include "xdmf_read.h"
#include "xdmf_utils.h"
#include <dolfinx/graph/Partitioning.h>
#include <dolfinx/mesh/MeshTags.h>

namespace dolfinx
{

namespace io
{
namespace xdmf_meshtags
{

/// TODO
template <typename T>
void add_meshtags(MPI_Comm comm, const mesh::MeshTags<T>& meshtags,
                  pugi::xml_node& xml_node, const hid_t h5_id,
                  const std::string name)
{
  // Get mesh
  assert(meshtags.mesh());
  std::shared_ptr<const mesh::Mesh> mesh = meshtags.mesh();
  const int dim = meshtags.dim();

  const std::vector<std::int32_t>& active_entities = meshtags.indices();

  const std::string path_prefix = "/MeshTags/" + name;
  xdmf_mesh::add_topology_data(comm, xml_node, h5_id, path_prefix,
                               mesh->topology(), mesh->geometry(), dim,
                               active_entities);

  // Add attribute node with values
  pugi::xml_node attribute_node = xml_node.append_child("Attribute");
  assert(attribute_node);
  attribute_node.append_attribute("Name") = name.c_str();
  attribute_node.append_attribute("AttributeType") = "Scalar";
  attribute_node.append_attribute("Center") = "Cell";

  const std::int64_t global_num_values
      = dolfinx::MPI::sum(comm, (std::int64_t)active_entities.size());

  const std::int64_t offset
      = dolfinx::MPI::global_offset(comm, active_entities.size(), true);
  const bool use_mpi_io = (dolfinx::MPI::size(comm) > 1);

  xdmf_utils::add_data_item(attribute_node, h5_id, path_prefix + "/Values",
                            meshtags.values(), offset, {global_num_values, 1},
                            "", use_mpi_io);
}

/// TODO
template <typename T>
mesh::MeshTags<T> read_meshtags(MPI_Comm comm,
                                const std::shared_ptr<const mesh::Mesh>& mesh,
                                pugi::xml_node& grid_node, const hid_t h5_id)
{

  pugi::xml_node topology_node = grid_node.child("Topology");

  // Get topology dataset node
  pugi::xml_node topology_data_node = topology_node.child("DataItem");
  const std::vector<std::int64_t> tdims
      = xdmf_utils::get_dataset_shape(topology_data_node);
  const int nnodes_per_entity = tdims[1];

  // Read topology data
  const std::vector<std::int64_t> topology_data
      = xdmf_read::get_dataset<std::int64_t>(comm, topology_data_node, h5_id);
  const std::int32_t num_local_file_entities
      = topology_data.size() / nnodes_per_entity;

  std::vector<std::int64_t> topo_unique;
  topo_unique.assign(topology_data.begin(), topology_data.end());

  std::sort(topo_unique.begin(), topo_unique.end());
  topo_unique.erase(std::unique(topo_unique.begin(), topo_unique.end()),
                    topo_unique.end());

  // Fetch cell type of meshtags and deduce its dimension
  const auto cell_type_str = xdmf_utils::get_cell_type(topology_node);
  const mesh::CellType cell_type = mesh::to_type(cell_type_str.first);
  const int e_dim = mesh::cell_dim(cell_type);

  const int dim = mesh->topology().dim();
  auto e_to_v = mesh->topology().connectivity(e_dim, 0);
  assert(e_to_v);
  auto e_to_c = mesh->topology().connectivity(e_dim, dim);
  assert(e_to_c);
  auto c_to_v = mesh->topology().connectivity(dim, 0);
  assert(c_to_v);

  const std::vector<std::int64_t>& igi
      = mesh->geometry().input_global_indices();

  //
  // Send input global indices to officers, based on input global index value
  //

  const std::int64_t num_igi_global = MPI::sum(comm, (std::int64_t)igi.size());

  // Split global array size and retrieve a range that this process/officer is
  // responsible for
  std::array<std::int64_t, 2> range = MPI::local_range(comm, num_igi_global);
  const int local_size = range[1] - range[0];

  const int comm_size = MPI::size(comm);
  std::vector<std::vector<std::int64_t>> send_igi(comm_size);
  std::vector<std::vector<std::int64_t>> recv_igi(comm_size);

  for (const auto gi : igi)
  {
    // TODO: Optimise this call
    // Figure out which process responsible for the input global index
    const int officer = MPI::index_owner(comm_size, gi, num_igi_global);
    send_igi[officer].push_back(gi);
  }

  MPI::all_to_all(comm, send_igi, recv_igi);

  //
  // Handle received input global indices, i.e. put the owner of it to
  // a global position, which is its value
  //

  std::vector<int> owners(local_size, 0);
  const std::size_t offset = MPI::global_offset(comm, local_size, true);

  for (int i = 0; i < comm_size; ++i)
  {
    const int num_recv_igi = (int)recv_igi[i].size();
    for (int j = 0; j < num_recv_igi; ++j)
    {
      const int local_index = recv_igi[i][j] - offset;
      assert(local_size > local_index);
      assert(local_index >= 0);
      owners[local_index] = i;
    }
  }

  //
  // Distribute the owners of input global indices
  //

  Eigen::Map<Eigen::Array<int, Eigen::Dynamic, 1>> owners_arr(owners.data(),
                                                              owners.size());

  // Distribute owners and fetch owners for the input global indices read from
  // file, i.e. for the unique topology data in file
  const Eigen::Array<int, Eigen::Dynamic, 1> dist_owners_arr
      = graph::Partitioning::distribute_data<int>(comm, topo_unique,
                                                  owners_arr);

  //
  // Figure out which process needs input global indices read from file
  // and send to it
  //

  // Prepare an array where on n-th position is the owner of n-th node
  std::vector<int> topo_owners(topology_data.size());
  for (std::size_t i = 0; i < topo_unique.size(); ++i)
    topo_owners[topo_unique[i]] = dist_owners_arr(i, 0);

  std::vector<std::vector<std::int64_t>> send_ents(comm_size);
  std::vector<std::vector<std::int64_t>> recv_ents(comm_size);
  std::vector<std::vector<T>> send_vals(comm_size);
  std::vector<std::vector<T>> recv_vals(comm_size);

  pugi::xml_node values_data_node
      = grid_node.child("Attribute").child("DataItem");

  std::vector<T> values
      = xdmf_read::get_dataset<T>(comm, values_data_node, h5_id);

  for (Eigen::Index e = 0; e < num_local_file_entities; ++e)
  {
    std::vector<std::int64_t> entity(nnodes_per_entity);
    std::vector<bool> sent(comm_size, false);

    for (int i = 0; i < nnodes_per_entity; ++i)
      entity[i] = topology_data[e * nnodes_per_entity + i];

    for (int i = 0; i < nnodes_per_entity; ++i)
    {
      // Entity could have as many owners as there are owners
      // of its nodes
      const int send_to = topo_owners[entity[i]];
      assert(send_to >= 0);
      if (!sent[send_to])
      {
        send_ents[send_to].insert(send_ents[send_to].end(), entity.begin(),
                                  entity.end());
        send_vals[send_to].push_back(values[e]);
        sent[send_to] = true;
      }
    }
  }

  MPI::all_to_all(comm, send_ents, recv_ents);
  MPI::all_to_all(comm, send_vals, recv_vals);

  //
  // Using just the information on current local mesh partition
  // prepare a mapping from *ordered* nodes of entity input global indices to
  // entity local index
  //

  std::map<std::vector<std::int64_t>, std::int32_t> entities_igi;

  auto map_e = mesh->topology().index_map(e_dim);
  assert(map_e);
  const std::int32_t num_entities = map_e->size_local() + map_e->num_ghosts();

  const graph::AdjacencyList<std::int32_t>& cells_g = mesh->geometry().dofmap();
  const std::vector<std::uint8_t> vtk_perm
      = cells::vtk_to_dolfin(cell_type, nnodes_per_entity);

  for (std::int32_t e = 0; e < num_entities; ++e)
  {
    std::vector<std::int64_t> entity_igi(nnodes_per_entity);

    // Iterate over all entities of the mesh
    // Find cell attached to the entity
    std::int32_t c = e_to_c->links(e)[0];
    auto cell_nodes = cells_g.links(c);
    auto cell_vertices = c_to_v->links(c);
    auto entity_vertices = e_to_v->links(e);

    for (int v = 0; v < entity_vertices.rows(); ++v)
    {
      // Find local index of vertex wrt. cell
      const int vertex = entity_vertices[vtk_perm[v]];
      auto it = std::find(cell_vertices.data(),
                          cell_vertices.data() + cell_vertices.rows(), vertex);
      assert(it != (cell_vertices.data() + cell_vertices.rows()));
      const int local_cell_vertex = std::distance(cell_vertices.data(), it);

      // Insert input global index for the node of the entitity
      entity_igi[v] = igi[cell_nodes[local_cell_vertex]];
    }

    // Sorting is needed to match with entities stored in file
    std::sort(entity_igi.begin(), entity_igi.end());
    entities_igi.insert({entity_igi, e});
  }

  //
  // Iterate over all received entities and find it in entities of
  // the mesh
  //

  std::vector<std::int32_t> indices;
  values.clear();

  for (int i = 0; i < comm_size; ++i)
  {
    const int num_recv_ents = (int)(recv_ents[i].size() / nnodes_per_entity);
    for (int e = 0; e < num_recv_ents; ++e)
    {
      std::vector<std::int64_t> _entity(&recv_ents[i][nnodes_per_entity * e],
                                        &recv_ents[i][nnodes_per_entity * e]
                                            + nnodes_per_entity);

      std::sort(_entity.begin(), _entity.end());

      const auto it = entities_igi.find(_entity);
      if (it != entities_igi.end())
      {
        indices.push_back(it->second);
        values.push_back(recv_vals[i][e]);
      }
    }
  }

  return mesh::MeshTags<T>(mesh, e_dim, indices, values);
}

} // namespace xdmf_meshtags
} // namespace io
} // namespace dolfinx