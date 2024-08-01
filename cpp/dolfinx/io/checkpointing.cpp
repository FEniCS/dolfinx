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
#include <dolfinx/mesh/utils.h>
#include <mpi.h>

/// @file checkpointing.h
/// @brief ADIOS2 based checkpointing
namespace
{
std::map<basix::element::lagrange_variant, std::string> variant_to_string{
    {basix::element::lagrange_variant::unset, "unset"},
    {basix::element::lagrange_variant::equispaced, "equispaced"},
    {basix::element::lagrange_variant::gll_warped, "gll_warped"},
};

std::map<std::string, basix::element::lagrange_variant> string_to_variant{
    {"unset", basix::element::lagrange_variant::unset},
    {"equispaced", basix::element::lagrange_variant::equispaced},
    {"gll_warped", basix::element::lagrange_variant::gll_warped},
};

} // namespace

namespace dolfinx::io::checkpointing
{
//-----------------------------------------------------------------------------
template <std::floating_point T>
void write_mesh(adios2::IO& io, adios2::Engine& engine,
                dolfinx::mesh::Mesh<T>& mesh)
{

  const mesh::Geometry<T>& geometry = mesh.geometry();
  std::shared_ptr<const mesh::Topology> topology = mesh.topology();

  // Variables/attributes to save
  std::int32_t dim = geometry.dim();
  std::uint64_t num_vertices_global, offset, num_cells_global, cell_offset;
  std::uint32_t num_vertices_local, num_cells_local, num_dofs_per_cell, degree;
  std::string cell_type, lagrange_variant;
  std::vector<std::int64_t> array_global;
  std::vector<std::int32_t> offsets_global;

  std::shared_ptr<const common::IndexMap> geom_imap;

  // Vertices information
  {
    geom_imap = mesh.geometry().index_map();
    num_vertices_global = geom_imap->size_global();
    num_vertices_local = geom_imap->size_local();
    offset = geom_imap->local_range()[0];
  }

  // Cells information
  {
    const std::shared_ptr<const common::IndexMap> topo_imap
        = topology->index_map(dim);
    num_cells_global = topo_imap->size_global();
    num_cells_local = topo_imap->size_local();
    cell_offset = topo_imap->local_range()[0];
  }

  // Coordinate element information
  {
    const fem::CoordinateElement<T>& cmap = mesh.geometry().cmap();
    fem::ElementDofLayout geom_layout = cmap.create_dof_layout();
    num_dofs_per_cell = geom_layout.num_entity_closure_dofs(dim);
    cell_type = mesh::to_string(cmap.cell_shape());
    degree = cmap.degree();
    lagrange_variant = variant_to_string[cmap.variant()];
  }

  const std::vector<int64_t> input_global_indices
      = geometry.input_global_indices();
  const std::span<const int64_t> input_global_indices_span(
      input_global_indices.begin(), num_vertices_local);
  const std::span<const T> mesh_x = geometry.x();

  // Connectivity
  {
    std::shared_ptr<const graph::AdjacencyList<std::int32_t>> connectivity
        = topology->connectivity(dim, 0);
    const std::vector<std::int32_t>& array = connectivity->array();
    const std::vector<std::int32_t>& offsets = connectivity->offsets();

    const std::span<const int32_t> array_span(array.begin(),
                                              offsets[num_cells_local]);

    array_global.resize(offsets[num_cells_local]);
    offsets_global.resize(num_cells_local + 1);

    std::iota(array_global.begin(), array_global.end(), 0);

    geom_imap->local_to_global(array_span, array_global);

    for (std::size_t i = 0; i < num_cells_local + 1; ++i)
      offsets_global[i] = offsets[i] + cell_offset * num_dofs_per_cell;
  }

  // ADIOS2 write attributes and variables
  {
    io.DefineAttribute<std::string>("name", mesh.name);
    io.DefineAttribute<std::int32_t>("dim", dim);
    io.DefineAttribute<std::string>("cell_type", cell_type);
    io.DefineAttribute<std::int32_t>("degree", degree);
    io.DefineAttribute<std::string>("lagrange_variant", lagrange_variant);

    adios2::Variable<std::uint64_t> var_num_vertices
        = io.DefineVariable<std::uint64_t>("num_vertices");
    adios2::Variable<std::uint64_t> var_num_cells
        = io.DefineVariable<std::uint64_t>("num_cells");
    adios2::Variable<std::uint32_t> var_num_dofs_per_cell
        = io.DefineVariable<std::uint32_t>("num_dofs_per_cell");

    adios2::Variable<std::int64_t> var_input_global_indices
        = io.DefineVariable<std::int64_t>(
            "input_global_indices", {num_vertices_global}, {offset},
            {num_vertices_local}, adios2::ConstantDims);

    adios2::Variable<T> var_x
        = io.DefineVariable<T>("x", {num_vertices_global, 3}, {offset, 0},
                               {num_vertices_local, 3}, adios2::ConstantDims);

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
    engine.Put(var_num_vertices, num_vertices_global);
    engine.Put(var_num_cells, num_cells_global);
    engine.Put(var_num_dofs_per_cell, num_dofs_per_cell);
    engine.Put(var_input_global_indices, input_global_indices_span.data());
    engine.Put(var_x, mesh_x.subspan(0, num_vertices_local * 3).data());
    engine.Put(var_topology_array, array_global.data());
    engine.Put(var_topology_offsets, offsets_global.data());
    engine.EndStep();
  }
}

//-----------------------------------------------------------------------------
/// @cond
template void write_mesh<float>(adios2::IO& io, adios2::Engine& engine,
                                dolfinx::mesh::Mesh<float>& mesh);

template void write_mesh<double>(adios2::IO& io, adios2::Engine& engine,
                                 dolfinx::mesh::Mesh<double>& mesh);

/// @endcond

//-----------------------------------------------------------------------------
template <std::floating_point T>
dolfinx::mesh::Mesh<T> read_mesh(adios2::IO& io, adios2::Engine& engine,
                                 MPI_Comm comm)
{

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  engine.BeginStep();

  // Attributes
  std::string name;
  std::int32_t dim, degree;
  mesh::CellType cell_type;
  basix::element::lagrange_variant lagrange_variant;

  // Read attributes
  {
    adios2::Attribute<std::string> var_name
        = io.InquireAttribute<std::string>("name");
    adios2::Attribute<std::int32_t> var_dim
        = io.InquireAttribute<std::int32_t>("dim");
    adios2::Attribute<std::string> var_cell_type
        = io.InquireAttribute<std::string>("cell_type");
    adios2::Attribute<std::int32_t> var_degree
        = io.InquireAttribute<std::int32_t>("degree");
    adios2::Attribute<std::string> var_variant
        = io.InquireAttribute<std::string>("lagrange_variant");

    name = var_name.Data()[0];
    dim = var_dim.Data()[0];
    cell_type = mesh::to_type(var_cell_type.Data()[0]);
    degree = var_degree.Data()[0];
    lagrange_variant = string_to_variant[var_variant.Data()[0]];
  }

  // Scalar variables
  std::uint64_t num_vertices_global;
  std::uint64_t num_cells_global;
  std::uint32_t num_dofs_per_cell;

  // Read scalar variables
  {
    adios2::Variable<std::uint64_t> var_num_vertices
        = io.InquireVariable<std::uint64_t>("num_vertices");
    adios2::Variable<std::uint64_t> var_num_cells
        = io.InquireVariable<std::uint64_t>("num_cells");
    adios2::Variable<std::uint32_t> var_num_dofs_per_cell
        = io.InquireVariable<std::uint32_t>("num_dofs_per_cell");

    engine.Get(var_num_vertices, num_vertices_global);
    engine.Get(var_num_cells, num_cells_global);
    engine.Get(var_num_dofs_per_cell, num_dofs_per_cell);
  }

  // Compute local sizes, offsets
  std::array<std::int64_t, 2> _local_range
      = dolfinx::MPI::local_range(rank, num_vertices_global, size);

  std::array<std::uint64_t, 2> local_range{(std::uint64_t)_local_range[0],
                                           (std::uint64_t)_local_range[1]};
  std::uint64_t num_vertices_local = local_range[1] - local_range[0];

  std::array<std::int64_t, 2> _cell_range
      = dolfinx::MPI::local_range(rank, num_cells_global, size);

  std::array<std::uint64_t, 2> cell_range{(std::uint64_t)_cell_range[0],
                                          (std::uint64_t)_cell_range[1]};
  std::uint64_t num_cells_local = cell_range[1] - cell_range[0];

  std::vector<int64_t> input_global_indices(num_vertices_local);
  std::vector<T> x(num_vertices_local * 3);
  std::vector<int64_t> array(num_cells_local * num_dofs_per_cell);
  std::vector<int32_t> offsets(num_cells_local + 1);

  {
    adios2::Variable<std::int64_t> var_input_global_indices
        = io.InquireVariable<std::int64_t>("input_global_indices");

    adios2::Variable<T> var_x = io.InquireVariable<T>("x");

    adios2::Variable<std::int64_t> var_topology_array
        = io.InquireVariable<std::int64_t>("topology_array");

    adios2::Variable<std::int32_t> var_topology_offsets
        = io.InquireVariable<std::int32_t>("topology_offsets");

    if (var_input_global_indices)
    {
      var_input_global_indices.SetSelection(
          {{local_range[0]}, {num_vertices_local}});
      engine.Get(var_input_global_indices, input_global_indices.data(),
                 adios2::Mode::Deferred);
    }

    if (var_x)
    {
      var_x.SetSelection({{local_range[0], 0}, {num_vertices_local, 3}});
      engine.Get(var_x, x.data(), adios2::Mode::Deferred);
    }

    if (var_topology_array)
    {
      var_topology_array.SetSelection({{cell_range[0] * num_dofs_per_cell},
                                       {cell_range[1] * num_dofs_per_cell}});
      engine.Get(var_topology_array, array.data(), adios2::Mode::Deferred);
    }

    if (var_topology_offsets)
    {
      var_topology_offsets.SetSelection({{cell_range[0]}, {cell_range[1] + 1}});
      engine.Get(var_topology_offsets, offsets.data(), adios2::Mode::Deferred);
    }

    engine.EndStep();

    std::int32_t cell_offset = offsets[0];
    for (auto offset = offsets.begin(); offset != offsets.end(); ++offset)
      *offset -= cell_offset;
  }

  std::vector<T> x_reduced(num_vertices_local * dim);
  for (std::uint32_t i = 0; i < num_vertices_local; ++i)
  {
    for (std::uint32_t j = 0; j < (std::uint32_t)dim; ++j)
      x_reduced[i * dim + j] = x[i * 3 + j];
  }

  fem::CoordinateElement<T> element
      = fem::CoordinateElement<T>(cell_type, degree, lagrange_variant);

  std::array<std::size_t, 2> xshape = {num_vertices_local, (std::uint32_t)dim};
  auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);

  mesh::Mesh<T> mesh = mesh::create_mesh(comm, comm, array, element, comm,
                                         x_reduced, xshape, part);
  return mesh;
}

//-----------------------------------------------------------------------------
/// @cond
template dolfinx::mesh::Mesh<float>
read_mesh<float>(adios2::IO& io, adios2::Engine& engine, MPI_Comm comm);

template dolfinx::mesh::Mesh<double>
read_mesh<double>(adios2::IO& io, adios2::Engine& engine, MPI_Comm comm);

/// @endcond

} // namespace dolfinx::io::checkpointing

#endif
