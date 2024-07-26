// Copyright (C) 2024 Abdullah Mujahid
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_ADIOS2

#include "checkpointing.h"
#include <adios2.h>
#include <basix/finite-element.h>
#include <dolfinx/mesh/Mesh.h>
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
void _write(MPI_Comm comm, std::string filename, std::string tag,
            // adios2::IO io, adios2::Engine engine,
            std::shared_ptr<dolfinx::mesh::Mesh<T>> mesh)
{
  adios2::ADIOS adios(comm);
  adios2::IO io = adios.DeclareIO(tag);
  adios2::Engine writer = io.Open(filename, adios2::Mode::Write);

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

  io.DefineAttribute<std::string>("name", mesh->name);
  io.DefineAttribute<std::int16_t>("dim", geometry.dim());
  io.DefineAttribute<std::string>("cell_type",
                                  dolfinx::mesh::to_string(cmap.cell_shape()));
  io.DefineAttribute<std::int32_t>("degree", cmap.degree());
  io.DefineAttribute<std::string>("lagrange_variant",
                                  lagrange_variants[cmap.variant()]);

  adios2::Variable<std::uint64_t> var_num_nodes
      = io.DefineVariable<std::uint64_t>("num_nodes");
  adios2::Variable<std::uint64_t> var_num_cells
      = io.DefineVariable<std::uint64_t>("num_cells");
  adios2::Variable<std::uint32_t> var_num_dofs_per_cell
      = io.DefineVariable<std::uint32_t>("num_dofs_per_cell");

  adios2::Variable<std::int64_t> var_input_global_indices
      = io.DefineVariable<std::int64_t>(
          "input_global_indices", {num_nodes_global}, {offset},
          {num_nodes_local}, adios2::ConstantDims);

  adios2::Variable<T> var_x
      = io.DefineVariable<T>("x", {num_nodes_global, 3}, {offset, 0},
                             {num_nodes_local, 3}, adios2::ConstantDims);

  adios2::Variable<std::int64_t> var_topology_array
      = io.DefineVariable<std::int64_t>(
          "topology_array", {num_cells_global * num_dofs_per_cell},
          {cell_offset * num_dofs_per_cell},
          {num_cells_local * num_dofs_per_cell}, adios2::ConstantDims);

  adios2::Variable<std::int32_t> var_topology_offsets
      = io.DefineVariable<std::int32_t>(
          "topology_offsets", {num_cells_global + 1}, {cell_offset},
          {num_cells_local + 1}, adios2::ConstantDims);

  writer.BeginStep();
  writer.Put(var_num_nodes, num_nodes_global);
  writer.Put(var_num_cells, num_cells_global);
  writer.Put(var_num_dofs_per_cell, num_dofs_per_cell);
  writer.Put(var_input_global_indices, input_global_indices_span.data());
  writer.Put(var_x, mesh_x.subspan(0, num_nodes_local * 3).data());
  writer.Put(var_topology_array, topology_array_global.data());
  writer.Put(var_topology_offsets, topology_offsets_span.data());
  writer.EndStep();
  writer.Close();
}

} // namespace

using namespace dolfinx::io::checkpointing;

//-----------------------------------------------------------------------------
void dolfinx::io::checkpointing::write(
    MPI_Comm comm, std::string filename, std::string tag,
    // adios2::IO io, adios2::Engine engine,
    std::shared_ptr<dolfinx::mesh::Mesh<float>> mesh)
{

  //   _write(io, engine, mesh);
  _write(comm, filename, tag, mesh);
}

//-----------------------------------------------------------------------------
void dolfinx::io::checkpointing::write(
    MPI_Comm comm, std::string filename, std::string tag,
    // adios2::IO io, adios2::Engine engine,
    std::shared_ptr<dolfinx::mesh::Mesh<double>> mesh)
{

  //   _write(io, engine, mesh);
  _write(comm, filename, tag, mesh);
}

#endif
