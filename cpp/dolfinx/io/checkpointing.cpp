// Copyright (C) 2024 Abdullah Mujahid
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_ADIOS2

#include "checkpointing.h"
#include "ADIOS2_utils.h"
#include <adios2.h>
#include <basix/finite-element.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/mesh/utils.h>
#include <mpi.h>

using namespace dolfinx;
using namespace dolfinx::io;

/// @file checkpointing.h
/// @brief ADIOS2 based checkpointing
namespace
{
std::map<basix::element::lagrange_variant, std::string> lagrange_variants{
    {basix::element::lagrange_variant::unset, "unset"},
    {basix::element::lagrange_variant::equispaced, "equispaced"},
    {basix::element::lagrange_variant::gll_warped, "gll_warped"},
};

template <std::floating_point T>
void _write_mesh(ADIOS2Engine& adios2engine,
                 std::shared_ptr<dolfinx::mesh::Mesh<T>> mesh)
{

  auto io = adios2engine.io();
  auto writer = adios2engine.engine();

  const dolfinx::mesh::Geometry<T>& geometry = mesh->geometry();
  auto topology = mesh->topology();

  std::int16_t mesh_dim = geometry.dim();

  auto imap = mesh->geometry().index_map();
  std::uint64_t num_nodes_global = imap->size_global();
  std::uint32_t num_nodes_local = imap->size_local();
  std::uint64_t offset = imap->local_range()[0];

  const std::shared_ptr<const dolfinx::common::IndexMap> topo_imap
      = topology->index_map(mesh_dim);
  std::uint64_t num_cells_global = topo_imap->size_global();
  std::uint32_t num_cells_local = topo_imap->size_local();
  std::uint64_t cell_offset = topo_imap->local_range()[0];

  auto cmap = mesh->geometry().cmap();
  auto geom_layout = cmap.create_dof_layout();
  std::uint32_t num_dofs_per_cell
      = geom_layout.num_entity_closure_dofs(mesh_dim);

  const std::vector<int64_t> input_global_indices
      = geometry.input_global_indices();
  const std::span<const int64_t> input_global_indices_span(
      input_global_indices.begin(), num_nodes_local);
  const std::span<const T> mesh_x = geometry.x();

  auto connectivity = topology->connectivity(mesh_dim, 0);
  auto topology_array = connectivity->array();
  auto topology_offsets = connectivity->offsets();

  const std::span<const int32_t> topology_array_span(
      topology_array.begin(), topology_offsets[num_cells_local]);
  std::vector<std::int64_t> topology_array_global(
      topology_offsets[num_cells_local]);

  std::iota(topology_array_global.begin(), topology_array_global.end(), 0);

  imap->local_to_global(topology_array_span, topology_array_global);

  for (std::size_t i = 0; i < num_cells_local + 1; ++i)
    topology_offsets[i] += cell_offset * num_dofs_per_cell;

  const std::span<const int32_t> topology_offsets_span(topology_offsets.begin(),
                                                       num_cells_local + 1);

  io->DefineAttribute<std::string>("name", mesh->name);
  io->DefineAttribute<std::int16_t>("dim", geometry.dim());
  io->DefineAttribute<std::string>("cell_type",
                                   dolfinx::mesh::to_string(cmap.cell_shape()));
  io->DefineAttribute<std::int32_t>("degree", cmap.degree());
  io->DefineAttribute<std::string>("lagrange_variant",
                                   lagrange_variants[cmap.variant()]);

  adios2::Variable<std::uint64_t> var_num_nodes
      = io->DefineVariable<std::uint64_t>("num_nodes");
  adios2::Variable<std::uint64_t> var_num_cells
      = io->DefineVariable<std::uint64_t>("num_cells");
  adios2::Variable<std::uint32_t> var_num_dofs_per_cell
      = io->DefineVariable<std::uint32_t>("num_dofs_per_cell");

  adios2::Variable<std::int64_t> var_input_global_indices
      = io->DefineVariable<std::int64_t>(
          "input_global_indices", {num_nodes_global}, {offset},
          {num_nodes_local}, adios2::ConstantDims);

  adios2::Variable<T> var_x
      = io->DefineVariable<T>("x", {num_nodes_global, 3}, {offset, 0},
                              {num_nodes_local, 3}, adios2::ConstantDims);

  adios2::Variable<std::int64_t> var_topology_array
      = io->DefineVariable<std::int64_t>(
          "topology_array", {num_cells_global * num_dofs_per_cell},
          {cell_offset * num_dofs_per_cell},
          {num_cells_local * num_dofs_per_cell}, adios2::ConstantDims);

  adios2::Variable<std::int32_t> var_topology_offsets
      = io->DefineVariable<std::int32_t>(
          "topology_offsets", {num_cells_global + 1}, {cell_offset},
          {num_cells_local + 1}, adios2::ConstantDims);

  writer->BeginStep();
  writer->Put(var_num_nodes, num_nodes_global);
  writer->Put(var_num_cells, num_cells_global);
  writer->Put(var_num_dofs_per_cell, num_dofs_per_cell);
  writer->Put(var_input_global_indices, input_global_indices_span.data());
  writer->Put(var_x, mesh_x.subspan(0, num_nodes_local * 3).data());
  writer->Put(var_topology_array, topology_array_global.data());
  writer->Put(var_topology_offsets, topology_offsets_span.data());
  writer->EndStep();
}

template <std::floating_point T>
void _write_meshtags(ADIOS2Engine& adios2engine,
                     std::shared_ptr<dolfinx::mesh::Mesh<T>> mesh,
                     std::shared_ptr<dolfinx::mesh::MeshTags<T>> meshtags)
{
  auto io = adios2engine.io();
  auto writer = adios2engine.engine();

  auto geometry = mesh->geometry();
  auto topology = mesh->topology();

  // meshtagsdata
  auto tag_entities = meshtags->indices();
  auto dim = meshtags->dim();

  auto cmap = mesh->geometry().cmap();
  auto geom_layout = cmap.create_dof_layout();
  std::uint32_t num_dofs_per_entity = geom_layout.num_entity_closure_dofs(dim);

  std::uint32_t num_tag_entities_local
      = meshtags->topology()->index_map(dim)->size_local();

  int num_tag_entities = tag_entities.size();

  std::vector<std::int32_t> local_tag_entities;
  local_tag_entities.reserve(num_tag_entities);

  std::uint64_t num_saved_tag_entities = 0;
  for (int i = 0; i < num_tag_entities; ++i)
  {
    if (tag_entities[i] < (int)num_tag_entities_local)
    {
      num_saved_tag_entities += 1;
      local_tag_entities.push_back(tag_entities[i]);
    }
  }
  local_tag_entities.resize(num_saved_tag_entities);

  // Compute the global offset for owned (local) vertex indices
  std::uint64_t local_start = 0;
  {
    MPI_Exscan(&num_saved_tag_entities, &local_start, 1, MPI_UINT64_T, MPI_SUM,
               mesh->comm());
  }

  std::uint64_t num_tag_entities_global = 0;
  MPI_Allreduce(&num_saved_tag_entities, &num_tag_entities_global, 1,
                MPI_UINT64_T, MPI_SUM, mesh->comm());

  auto values = meshtags->values();
  const std::span<const T> local_values(values.begin(), num_saved_tag_entities);

  std::vector<std::int32_t> entities_to_geometry
      = mesh::entities_to_geometry(*mesh, dim, tag_entities, false);

  auto imap = mesh->geometry().index_map();
  std::vector<std::int64_t> topology_array(entities_to_geometry.size());

  std::iota(topology_array.begin(), topology_array.end(), 0);

  imap->local_to_global(entities_to_geometry, topology_array);

  std::string name = meshtags->name;

  io->DefineAttribute<std::string>("meshtags_name", meshtags->name);
  io->DefineAttribute<std::int16_t>("meshtags_dim", dim);

  adios2::Variable<std::uint64_t> var_num_tag_entities_global
      = io->DefineVariable<std::uint64_t>("num_tag_entities_global");
  adios2::Variable<std::uint32_t> var_num_dofs_per_entity
      = io->DefineVariable<std::uint32_t>("num_dofs_per_entity");

  adios2::Variable<std::int64_t> var_topology
      = io->DefineVariable<std::int64_t>(
          name + "_topology", {num_tag_entities_global, num_dofs_per_entity},
          {local_start, 0}, {num_tag_entities_local, num_dofs_per_entity},
          adios2::ConstantDims);

  adios2::Variable<T> var_values = io->DefineVariable<T>(
      name + "_values", {num_tag_entities_global}, {local_start},
      {num_saved_tag_entities}, adios2::ConstantDims);

  writer->BeginStep();
  writer->Put(var_num_dofs_per_entity, num_dofs_per_entity);
  writer->Put(var_num_tag_entities_global, num_tag_entities_global);
  writer->Put(var_topology, topology_array.data());
  writer->Put(var_values, local_values.data());
  writer->EndStep();
}

} // namespace

using namespace dolfinx::io::checkpointing;

//-----------------------------------------------------------------------------
void dolfinx::io::checkpointing::write_mesh(
    ADIOS2Engine& adios2engine,
    std::shared_ptr<dolfinx::mesh::Mesh<float>> mesh)
{

  _write_mesh(adios2engine, mesh);
}

//-----------------------------------------------------------------------------
void dolfinx::io::checkpointing::write_mesh(
    ADIOS2Engine& adios2engine,
    std::shared_ptr<dolfinx::mesh::Mesh<double>> mesh)
{

  _write_mesh(adios2engine, mesh);
}

//-----------------------------------------------------------------------------
void dolfinx::io::checkpointing::write_meshtags(
    ADIOS2Engine& adios2engine,
    std::shared_ptr<dolfinx::mesh::Mesh<float>> mesh,
    std::shared_ptr<dolfinx::mesh::MeshTags<float>> meshtags)
{

  _write_meshtags(adios2engine, mesh, meshtags);
}

//-----------------------------------------------------------------------------
void dolfinx::io::checkpointing::write_meshtags(
    ADIOS2Engine& adios2engine,
    std::shared_ptr<dolfinx::mesh::Mesh<double>> mesh,
    std::shared_ptr<dolfinx::mesh::MeshTags<double>> meshtags)
{

  _write_meshtags(adios2engine, mesh, meshtags);
}

#endif
