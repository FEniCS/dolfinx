// Copyright (C) year authors
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
            std::shared_ptr<dolfinx::mesh::Mesh<T>> mesh)
{
  adios2::ADIOS adios(comm);
  adios2::IO io = adios.DeclareIO(tag);
  adios2::Engine writer = io.Open(filename, adios2::Mode::Write);

  const dolfinx::mesh::Geometry<T>& geometry = mesh->geometry();
  auto topology = mesh->topology();

  std::int16_t mesh_dim = geometry.dim();
  const std::vector<int64_t> mesh_input_global_indices
      = geometry.input_global_indices();
  const std::span<const int64_t> mesh_input_global_indices_span(
      mesh_input_global_indices.begin(), mesh_input_global_indices.end());
  const std::span<const T> mesh_x = geometry.x();

  auto imap = mesh->geometry().index_map();
  std::int64_t num_nodes_global = imap->size_global();
  std::int32_t num_nodes_local = imap->size_local();
  std::int64_t offset = imap->local_range()[0];

  const std::shared_ptr<const dolfinx::common::IndexMap> topo_imap
      = topology->index_map(mesh_dim);
  std::int64_t num_cells_global = topo_imap->size_global();
  std::int32_t num_cells_local = topo_imap->size_local();
  std::int64_t cell_offset = topo_imap->local_range()[0];

  auto cmap = mesh->geometry().cmap();
  auto geom_layout = cmap.create_dof_layout();
  auto num_dofs_per_cell = geom_layout.num_entity_closure_dofs(mesh_dim);

  io.DefineAttribute<std::string>("name", mesh->name);
  io.DefineAttribute<std::int16_t>("dim", geometry.dim());
  io.DefineAttribute<std::string>("CellType",
                                  dolfinx::mesh::to_string(cmap.cell_shape()));
  io.DefineAttribute<std::int32_t>("Degree", cmap.degree());
  io.DefineAttribute<std::string>("LagrangeVariant",
                                  lagrange_variants[cmap.variant()]);

  adios2::Variable<std::int64_t> n_nodes
      = io.DefineVariable<std::int64_t>("n_nodes");
  adios2::Variable<std::int64_t> n_cells
      = io.DefineVariable<std::int64_t>("n_cells");
  adios2::Variable<std::int32_t> n_dofs_per_cell
      = io.DefineVariable<std::int32_t>("n_dofs_per_cell");

  adios2::Variable<std::int64_t> input_global_indices
      = io.DefineVariable<std::int64_t>(
          "input_global_indices", {(long unsigned int)num_nodes_global},
          {(long unsigned int)offset}, {(long unsigned int)num_nodes_local},
          adios2::ConstantDims);

  adios2::Variable<T> x = io.DefineVariable<T>(
      "Points", {(long unsigned int)num_nodes_global, 3},
      {(long unsigned int)offset, 0}, {(long unsigned int)num_nodes_local, 3},
      adios2::ConstantDims);

  adios2::Variable<std::int64_t> cell_indices = io.DefineVariable<std::int64_t>(
      "cell_indices",
      {(long unsigned int)(num_cells_global * num_dofs_per_cell)},
      {(long unsigned int)(cell_offset * num_dofs_per_cell)},
      {(long unsigned int)(num_cells_local * num_dofs_per_cell)},
      adios2::ConstantDims);

  adios2::Variable<std::int32_t> cell_indices_offsets
      = io.DefineVariable<std::int32_t>(
          "cell_indices_offsets", {(long unsigned int)(num_cells_global + 1)},
          {(long unsigned int)(cell_offset)},
          {(long unsigned int)(num_cells_local + 1)}, adios2::ConstantDims);

  auto connectivity = topology->connectivity(mesh_dim, 0);
  auto indices = connectivity->array();
  const std::span<const int32_t> indices_span(indices.begin(), indices.end());

  auto indices_offsets = connectivity->offsets();
  for (std::size_t i = 0; i < indices_offsets.size(); ++i)
  {
    indices_offsets[i] += cell_offset * num_dofs_per_cell;
  }

  const std::span<const int32_t> indices_offsets_span(indices_offsets.begin(),
                                                      indices_offsets.end());

  std::vector<std::int64_t> connectivity_nodes_global(
      indices_offsets[num_cells_local]);

  imap->local_to_global(
      indices_span.subspan(0, indices_offsets[num_cells_local]),
      connectivity_nodes_global);

  writer.BeginStep();
  writer.Put(n_nodes, num_nodes_global);
  writer.Put(n_cells, num_cells_global);
  writer.Put(n_dofs_per_cell, num_dofs_per_cell);
  writer.Put(input_global_indices,
             mesh_input_global_indices_span.subspan(0, num_nodes_local).data());
  writer.Put(x, mesh_x.subspan(0, num_nodes_local * 3).data());
  writer.Put(cell_indices, connectivity_nodes_global.data());
  writer.Put(cell_indices_offsets,
             indices_offsets_span.subspan(0, num_cells_local + 1).data());
  writer.EndStep();
  writer.Close();
}

} // namespace

using namespace dolfinx::io::checkpointing;

//-----------------------------------------------------------------------------
void dolfinx::io::checkpointing::write(
    MPI_Comm comm, std::string filename, std::string tag,
    std::shared_ptr<dolfinx::mesh::Mesh<float>> mesh)
{

  _write(comm, filename, tag, mesh);
}

//-----------------------------------------------------------------------------
void dolfinx::io::checkpointing::write(
    MPI_Comm comm, std::string filename, std::string tag,
    std::shared_ptr<dolfinx::mesh::Mesh<double>> mesh)
{

  _write(comm, filename, tag, mesh);
}

#endif