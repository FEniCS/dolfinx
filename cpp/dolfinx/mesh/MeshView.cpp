// Copyright (C) 2021 Joseph Dean, Jørgen S. Dokken, Sarah Roggendorf
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
#include "MeshView.h"
#include <dolfinx/common/IndexMap.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
using namespace dolfinx;
using namespace dolfinx::mesh;

MeshView::MeshView(std::shared_ptr<const MeshTags<std::int32_t>> meshtag)
    : _mesh(meshtag->mesh()), _entities(meshtag->indices()),
      _dim(meshtag->dim()), _topology()
{

  // Create entity to vertex connectivity
  _mesh->topology_mutable().create_connectivity(_dim, 0);
  auto e_to_v = _mesh->topology().connectivity(_dim, 0);
  assert(e_to_v);

  // Create subset of unique vertices
  std::set<std::int32_t> view_vertices;
  for (auto entity : _entities)
  {
    auto e_vertices = e_to_v->links(entity);
    for (auto vertex : e_vertices)
      view_vertices.insert(vertex);
  }

  // Create index map for unique vertices
  // FIXME: Needs additional info for parallel
  auto vertex_imap = std::make_shared<const common::IndexMap>(
      _mesh->mpi_comm(), view_vertices.size());

  // Create index map for cells
  auto entity_imap = std::make_shared<const common::IndexMap>(_mesh->mpi_comm(),
                                                              _entities.size());

  // Create vertex map from mesh to mesh-view
  std::map<std::int32_t, std::int32_t> mesh_to_view_vertices;
  auto first_vertex = view_vertices.begin();
  for (std::int32_t i = 0; i < vertex_imap->size_local(); ++i)
    mesh_to_view_vertices.insert({*first_vertex++, i});

  // Create new local adjacency-list for cell dofs (used in topology creation),
  // using the new MeshView vertex-map
  std::vector<std::int32_t> entity_dofs_view;
  // FIXME: Add reserve here at some point
  std::vector<std::int32_t> entity_dofs_offset_view = {0};
  entity_dofs_offset_view.reserve(_entities.size() + 1);
  for (auto entity : _entities)
  {
    auto vertices = e_to_v->links(entity);
    for (std::size_t i = 0; i < vertices.size(); ++i)
      entity_dofs_view.push_back(mesh_to_view_vertices[vertices[i]]);
    entity_dofs_offset_view.push_back(entity_dofs_view.size());
  }
  // Create Adjacency lists
  graph::AdjacencyList<std::int32_t> connectivity(entity_dofs_view,
                                                  entity_dofs_offset_view);

  // Create topology
  CellType entity_type = cell_entity_type(_mesh->topology().cell_type(), _dim);
  _topology = std::make_shared<Topology>(_mesh->mpi_comm(), entity_type);

  // Set index maps (vertex map and cell map)
  _topology->set_index_map(0, vertex_imap);
  _topology->set_index_map(_dim, entity_imap);

  // Set connectivities vertex->vertex (identity) and cell->vertex
  auto v_to_v = std::make_shared<graph::AdjacencyList<std::int32_t>>(
      vertex_imap->size_local());
  _topology->set_connectivity(v_to_v, 0, 0);
  _topology->set_connectivity(
      std::make_shared<graph::AdjacencyList<std::int32_t>>(connectivity), _dim,
      0);
}
std::shared_ptr<mesh::Topology> MeshView::topology() { return _topology; }
std::shared_ptr<const Mesh> MeshView::mesh() { return _mesh; }
std::int32_t MeshView::dim() { return _dim; }
const std::vector<std::int32_t>& MeshView::entities() { return _entities; }
