// Copyright (C) 2021 Jorgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MeshView.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/common/utils.h>
#include <dolfinx/graph/AdjacencyList.h>

using namespace dolfinx;
using namespace dolfinx::mesh;

//-----------------------------------------------------------------------------
MeshView::MeshView(const std::shared_ptr<const Mesh> parent_mesh, int dim,
                   tcb::span<std::int32_t> entities)
    : _parent_mesh(parent_mesh), _parent_entity_map(), _parent_vertex_map(),
      _dim(dim), _topology()
{
  _parent_entity_map.reserve(entities.size());

  Topology& parent_topology = _parent_mesh->topology_mutable();
  parent_topology.create_connectivity(_dim, 0);

  // Extract all vertices used in view (local and ghosts)
  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> e_to_v
      = parent_topology.connectivity(_dim, 0);
  const CellType entity_type
      = mesh::cell_entity_type(_parent_mesh->topology().cell_type(), _dim);
  const int num_vertices_per_entity = mesh::cell_num_entities(entity_type, 0);
  std::vector<std::int32_t> view_vertices;
  view_vertices.reserve(entities.size() * num_vertices_per_entity);
  for (auto e : entities)
  {
    auto vertices = e_to_v->links(e);
    view_vertices.insert(view_vertices.end(), vertices.begin(), vertices.end());
  }
  // Create unique set of sorted vertices
  std::sort(view_vertices.begin(), view_vertices.end());
  view_vertices.erase(std::unique(view_vertices.begin(), view_vertices.end()),
                      view_vertices.end());

  // Compress vertex map
  std::shared_ptr<const common::IndexMap> vertex_map_org
      = _parent_mesh->topology().index_map(0);
  auto [vertex_map, global_parent_vertices]
      = dolfinx::common::compress_index_map(vertex_map_org, view_vertices);
  _parent_vertex_map.resize(global_parent_vertices.size());

  // Get mapping from local child vertices to local parent vertices
  vertex_map_org->global_to_local(global_parent_vertices, _parent_vertex_map);

  // Create compressed index map for the entities
  std::shared_ptr<const common::IndexMap> entity_map_org
      = _parent_mesh->topology().index_map(_dim);
  auto [entity_map, global_parent_entities]
      = dolfinx::common::compress_index_map(entity_map_org, entities);
  _parent_entity_map.resize(global_parent_entities.size());
  entity_map_org->global_to_local(global_parent_entities, _parent_entity_map);

  // Create inverse map (parent vertices->child vertices)
  const std::int32_t num_vertices_org
      = vertex_map_org->size_local() + vertex_map_org->num_ghosts();
  std::vector<std::int32_t> vertex_offset(num_vertices_org + 1);
  std::vector<std::int32_t> is_parent_vertex(num_vertices_org, 0);
  std::vector<std::int32_t> vertex_data(_parent_vertex_map.size());
  std::vector<std::int32_t> sort_index(_parent_vertex_map.size());
  std::iota(sort_index.begin(), sort_index.end(), 0);
  dolfinx::argsort_radix<std::int32_t, 16>(xtl::span(_parent_vertex_map),
                                           xtl::span(sort_index));
  for (auto parent_vertex : _parent_vertex_map)
    is_parent_vertex[parent_vertex]++;
  std::partial_sum(is_parent_vertex.begin(), is_parent_vertex.end(),
                   vertex_offset.begin() + 1);
  for (std::size_t i = 0; i < sort_index.size(); ++i)
    vertex_data[i] = sort_index[i];
  auto p_to_c_vertex
      = dolfinx::graph::AdjacencyList<std::int32_t>(vertex_data, vertex_offset);

  // Create new entity-vertex connectivity for child
  std::vector<std::int32_t> child_entities;
  child_entities.reserve(global_parent_entities.size()
                         * num_vertices_per_entity);
  std::vector<std::int32_t> offsets_e(1, 0);
  offsets_e.reserve(global_parent_entities.size() + 1);
  for (std::size_t i = 0; i < _parent_entity_map.size(); ++i)
  {
    auto vertices = e_to_v->links(_parent_entity_map[i]);
    for (auto vertex : vertices)
    {
      auto child_vertex = p_to_c_vertex.links(vertex);
      assert(child_vertex.size() == 1);
      child_entities.push_back(child_vertex[0]);
    }
    offsets_e.push_back(child_entities.size());
  }
  std::shared_ptr<graph::AdjacencyList<std::int32_t>> e_to_v_child
      = std::make_shared<graph::AdjacencyList<std::int32_t>>(child_entities,
                                                             offsets_e);

  // Create vertex to vertex map (is identity)
  std::shared_ptr<graph::AdjacencyList<std::int32_t>> v_to_v
      = std::make_shared<graph::AdjacencyList<std::int32_t>>(
          vertex_map->size_local() + vertex_map->num_ghosts());

  // Create topology for mesh view
  _topology
      = std::make_shared<mesh::Topology>(_parent_mesh->mpi_comm(), entity_type);
  _topology->set_index_map(0, vertex_map);
  _topology->set_index_map(_dim, entity_map);
  _topology->set_connectivity(v_to_v, 0, 0);
  _topology->set_connectivity(e_to_v_child, _dim, 0);
};