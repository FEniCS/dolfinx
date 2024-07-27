// Copyright (C) 2024 Abdullah Mujahid
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_ADIOS2

#include "checkpointing.h"
#include <adios2.h>
#include <basix/finite-element.h>
#include <dolfinx/fem/CoordinateElement.h>
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
} // namespace

namespace dolfinx::io::checkpointing
{
template <std::floating_point T>
void write_mesh(adios2::IO io, adios2::Engine engine,
                dolfinx::mesh::Mesh<T> mesh)
{

  const mesh::Geometry<T>& geometry = mesh.geometry();
  std::shared_ptr<const mesh::Topology> topology = mesh.topology();

  std::int32_t dim = geometry.dim();

  std::shared_ptr<const common::IndexMap> geom_imap
      = mesh.geometry().index_map();
  std::uint64_t num_nodes_global = geom_imap->size_global();
  std::uint32_t num_nodes_local = geom_imap->size_local();
  std::uint64_t offset = geom_imap->local_range()[0];

  const std::shared_ptr<const common::IndexMap> topo_imap
      = topology->index_map(dim);
  std::uint64_t num_cells_global = topo_imap->size_global();
  std::uint32_t num_cells_local = topo_imap->size_local();
  std::uint64_t cell_offset = topo_imap->local_range()[0];

  const fem::CoordinateElement<T>& cmap = mesh.geometry().cmap();
  fem::ElementDofLayout geom_layout = cmap.create_dof_layout();
  std::uint32_t num_dofs_per_cell = geom_layout.num_entity_closure_dofs(dim);

  const std::vector<int64_t> input_global_indices
      = geometry.input_global_indices();
  const std::span<const int64_t> input_global_indices_span(
      input_global_indices.begin(), num_nodes_local);
  const std::span<const T> mesh_x = geometry.x();

  std::shared_ptr<const graph::AdjacencyList<std::int32_t>> connectivity
      = topology->connectivity(dim, 0);
  const std::vector<std::int32_t>& array = connectivity->array();
  const std::vector<std::int32_t>& offsets = connectivity->offsets();

  const std::span<const int32_t> array_span(array.begin(),
                                            offsets[num_cells_local]);
  std::vector<std::int64_t> array_global(offsets[num_cells_local]);

  std::vector<std::int32_t> offsets_global(num_cells_local + 1);

  std::iota(array_global.begin(), array_global.end(), 0);

  geom_imap->local_to_global(array_span, array_global);

  for (std::size_t i = 0; i < num_cells_local + 1; ++i)
    offsets_global[i] = offsets[i] + cell_offset * num_dofs_per_cell;

  io.DefineAttribute<std::string>("name", mesh.name);
  io.DefineAttribute<std::int32_t>("dim", geometry.dim());
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

  engine.BeginStep();
  engine.Put(var_num_nodes, num_nodes_global);
  engine.Put(var_num_cells, num_cells_global);
  engine.Put(var_num_dofs_per_cell, num_dofs_per_cell);
  engine.Put(var_input_global_indices, input_global_indices_span.data());
  engine.Put(var_x, mesh_x.subspan(0, num_nodes_local * 3).data());
  engine.Put(var_topology_array, array_global.data());
  engine.Put(var_topology_offsets, offsets_global.data());
  engine.EndStep();
}

template void write_mesh<float>(adios2::IO io, adios2::Engine engine,
                                dolfinx::mesh::Mesh<float> mesh);

template void write_mesh<double>(adios2::IO io, adios2::Engine engine,
                                 dolfinx::mesh::Mesh<double> mesh);

} // namespace dolfinx::io::checkpointing

#endif
