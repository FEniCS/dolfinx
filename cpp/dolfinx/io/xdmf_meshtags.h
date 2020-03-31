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
                                pugi::xml_node& grid_node,
                                pugi::xml_node& flags_node, const hid_t h5_id)
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

  // Read flags
  const std::vector<std::int64_t> file_flags
      = xdmf_mesh::read_flags(comm, h5_id, flags_node);

  // Map flags vector into Eigen array for the use in distribute_data
  const Eigen::Map<const Eigen::Array<std::int64_t, Eigen::Dynamic, 1>>
      file_flags_arr(file_flags.data(), file_flags.size());

  // Extract only unique and sorted topology nodes
  // Sorting is needed for call to distribute_data
  // Uniqueness is to reduce the amount of data communicated
  std::vector<std::int64_t> topo_unique = topology_data;
  std::sort(topo_unique.begin(), topo_unique.end());
  topo_unique.erase(std::unique(topo_unique.begin(), topo_unique.end()),
                    topo_unique.end());

  // Distribute flags according to unique topology nodes
  const Eigen::Array<std::int64_t, Eigen::Dynamic, 1> dist_file_flags
      = graph::Partitioning::distribute_data<std::int64_t>(comm, topo_unique,
                                                           file_flags_arr);

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

  const std::vector<std::int64_t>& geom_flags = mesh->geometry().flags();

  //
  // Send flags to officers, based on flag's value
  //

  const std::int64_t num_flags_global
      = MPI::sum(comm, (std::int64_t)geom_flags.size());

  // Split global array size and retrieve a range that this process/officer is
  // responsible for
  std::array<std::int64_t, 2> range = MPI::local_range(comm, num_flags_global);
  const int local_size = range[1] - range[0];

  const int comm_size = MPI::size(comm);
  std::vector<std::vector<std::int64_t>> send_flags(comm_size);
  std::vector<std::vector<std::int64_t>> recv_flags(comm_size);

  for (auto flag : geom_flags)
  {
    // TODO: Optimise this call
    // Figure out which process responsible for the flag
    const int officer = MPI::index_owner(comm_size, flag, num_flags_global);
    send_flags[officer].push_back(flag);
  }

  MPI::all_to_all(comm, send_flags, recv_flags);

  //
  // Handle received flags, i.e. put the owner of the flag to
  // a global position, which is the value of the flag
  //

  std::vector<int> owners(local_size, -1);
  const std::size_t offset = MPI::global_offset(comm, local_size, true);

  for (int i = 0; i < comm_size; ++i)
  {
    const int num_recv_flags = (int)recv_flags[i].size();
    for (int j = 0; j < num_recv_flags; ++j)
    {
      const int local_index = recv_flags[i][j] - offset;
      assert(local_size > local_index);
      assert(local_index >= 0);
      owners[local_index] = i;
    }
  }

  //
  // Distribute the owners of flags
  //

  // Need to sort flags for the call to distribute_data
  // Store sorting permutation for later use on unique topology data
  std::vector<int> perm(dist_file_flags.rows());
  std::iota(perm.begin(), perm.end(), 0);
  std::sort(perm.begin(), perm.end(), [&](const int& a, const int& b) {
    return (dist_file_flags(a, 0) < dist_file_flags(b, 0));
  });

  // Apply the sorting permutation
  std::vector<std::int64_t> dist_flags_sorted;
  dist_flags_sorted.reserve(dist_file_flags.rows());

  for (int i = 0; i < dist_file_flags.rows(); ++i)
    dist_flags_sorted.push_back(dist_file_flags(perm[i], 0));

  Eigen::Map<Eigen::Array<int, Eigen::Dynamic, 1>> owners_arr(owners.data(),
                                                              owners.size());

  // Distribute owners and fetch owners for the flags read from file
  const Eigen::Array<int, Eigen::Dynamic, 1> dist_read_flags_owners_arr
      = graph::Partitioning::distribute_data<int>(comm, dist_flags_sorted,
                                                  owners_arr);

  //
  // Figure out which process needs flags read from file
  // and send to it
  //

  // Prepare a mapping (topology number in file: (flag, flag owner))
  std::unordered_map<std::int64_t, std::pair<std::int64_t, int>> topo_to_flags;
  for (std::size_t i = 0; i < topo_unique.size(); ++i)
  {
    topo_to_flags[topo_unique[perm[i]]]
        = {dist_flags_sorted[i], dist_read_flags_owners_arr(i, 0)};
  }

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
    std::vector<int> entity_owners(nnodes_per_entity);
    std::vector<bool> sent(comm_size, false);

    for (int i = 0; i < nnodes_per_entity; ++i)
      entity[i] = topo_to_flags[topology_data[e * nnodes_per_entity + i]].first;

    for (int i = 0; i < nnodes_per_entity; ++i)
    {
      // Entity could have as many owners as there are owners
      // of its flags
      const int send_to
          = topo_to_flags[topology_data[e * nnodes_per_entity + i]].second;
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
  // prepare a mapping from *ordered* nodes of entity flags to entity local
  // index
  //

  std::map<std::vector<std::int64_t>, std::int32_t> entities_flags;

  auto map_e = mesh->topology().index_map(e_dim);
  assert(map_e);
  const std::int32_t num_entities = map_e->size_local() + map_e->num_ghosts();

  const graph::AdjacencyList<std::int32_t>& cells_g = mesh->geometry().dofmap();
  const std::vector<std::uint8_t> vtk_perm
      = cells::vtk_to_dolfin(cell_type, nnodes_per_entity);

  for (std::int32_t e = 0; e < num_entities; ++e)
  {
    std::vector<std::int64_t> entity_flags(nnodes_per_entity);

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

      // Insert flag for the node of the entitity
      entity_flags[v] = geom_flags[cell_nodes[local_cell_vertex]];
    }

    // Sorting is needed to match with entities stored in file
    std::sort(entity_flags.begin(), entity_flags.end());
    entities_flags.insert({entity_flags, e});
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
      std::vector<std::int64_t> flags(&recv_ents[i][nnodes_per_entity * e],
                                      &recv_ents[i][nnodes_per_entity * e]
                                          + nnodes_per_entity);

      std::sort(flags.begin(), flags.end());

      const auto it = entities_flags.find(flags);
      if (it != entities_flags.end())
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