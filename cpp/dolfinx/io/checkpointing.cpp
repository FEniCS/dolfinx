// Copyright (C) 2024 Abdullah Mujahid, Jørgen S. Dokken, Jack S. Hale
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_ADIOS2

#include "checkpointing.h"
#include <adios2.h>
#include <basix/finite-element.h>
#include <dolfinx/common/defines.h>
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

namespace dolfinx::io::native
{
//-----------------------------------------------------------------------------
template <std::floating_point T>
void write_mesh(adios2::IO& io, adios2::Engine& engine,
                const dolfinx::mesh::Mesh<T>& mesh)
{

  const mesh::Geometry<T>& geometry = mesh.geometry();
  std::shared_ptr<const mesh::Topology> topology = mesh.topology();
  assert(topology);

  // Variables/attributes to save
  std::int32_t dim = geometry.dim();
  std::int32_t tdim = topology->dim();
  std::uint64_t num_nodes_global, offset, num_cells_global, cell_offset;
  std::uint32_t num_nodes_local, num_cells_local, num_dofs_per_cell, degree;
  std::string cell_type, lagrange_variant;
  std::vector<std::int64_t> array_global;
  std::vector<std::int32_t> offsets_global;

  std::shared_ptr<const common::IndexMap> geom_imap;

  // Nodes information
  {
    geom_imap = geometry.index_map();
    num_nodes_global = geom_imap->size_global();
    num_nodes_local = geom_imap->size_local();
    offset = geom_imap->local_range()[0];
  }

  // Cells information
  {
    const std::shared_ptr<const common::IndexMap> topo_imap
        = topology->index_map(tdim);
    assert(topo_imap);

    num_cells_global = topo_imap->size_global();
    num_cells_local = topo_imap->size_local();
    cell_offset = topo_imap->local_range()[0];
  }

  // Coordinate element information
  {
    const fem::CoordinateElement<T>& cmap = geometry.cmap();
    cell_type = mesh::to_string(cmap.cell_shape());
    degree = cmap.degree();
    lagrange_variant = variant_to_string[cmap.variant()];
  }

  const std::span<const T> mesh_x = geometry.x();

  // Connectivity
  {
    auto dofmap = geometry.dofmap();
    num_dofs_per_cell = dofmap.extent(1);
    std::vector<std::int32_t> connectivity;
    connectivity.reserve(num_cells_local * num_dofs_per_cell);
    for (std::size_t i = 0; i < num_cells_local; ++i)
      for (std::size_t j = 0; j < num_dofs_per_cell; ++j)
        connectivity.push_back(dofmap(i, j));

    array_global.resize(num_cells_local * num_dofs_per_cell);
    std::iota(array_global.begin(), array_global.end(), 0);

    geom_imap->local_to_global(connectivity, array_global);
    offsets_global.resize(num_cells_local + 1);
    for (std::size_t i = 0; i < num_cells_local + 1; ++i)
      offsets_global[i] = (i + cell_offset) * num_dofs_per_cell;
  }

  // ADIOS2 write attributes and variables
  {
    io.DefineAttribute<std::string>("version", dolfinx::version());
    io.DefineAttribute<std::string>("git_hash", dolfinx::git_commit_hash());
    io.DefineAttribute<std::string>("name", mesh.name);
    io.DefineAttribute<std::int32_t>("dim", dim);
    io.DefineAttribute<std::int32_t>("tdim", tdim);
    io.DefineAttribute<std::string>("cell_type", cell_type);
    io.DefineAttribute<std::int32_t>("degree", degree);
    io.DefineAttribute<std::string>("lagrange_variant", lagrange_variant);

    adios2::Variable<std::uint64_t> var_num_nodes
        = io.DefineVariable<std::uint64_t>("num_nodes");
    adios2::Variable<std::uint64_t> var_num_cells
        = io.DefineVariable<std::uint64_t>("num_cells");
    adios2::Variable<std::uint32_t> var_num_dofs_per_cell
        = io.DefineVariable<std::uint32_t>("num_dofs_per_cell");

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
    engine.Put(var_x, mesh_x.subspan(0, num_nodes_local * 3).data());
    engine.Put(var_topology_array, array_global.data());
    engine.Put(var_topology_offsets, offsets_global.data());
    engine.EndStep();
  }
}

//-----------------------------------------------------------------------------
/// @cond
template void write_mesh<float>(adios2::IO& io, adios2::Engine& engine,
                                const dolfinx::mesh::Mesh<float>& mesh);

template void write_mesh<double>(adios2::IO& io, adios2::Engine& engine,
                                 const dolfinx::mesh::Mesh<double>& mesh);

/// @endcond

//-----------------------------------------------------------------------------
template <std::floating_point T>
dolfinx::mesh::Mesh<T> read_mesh(adios2::IO& io, adios2::Engine& engine,
                                 MPI_Comm comm)
{

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // engine.BeginStep();

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
  std::uint64_t num_nodes_global;
  std::uint64_t num_cells_global;
  std::uint32_t num_dofs_per_cell;

  // Read scalar variables
  {
    adios2::Variable<std::uint64_t> var_num_nodes
        = io.InquireVariable<std::uint64_t>("num_nodes");
    adios2::Variable<std::uint64_t> var_num_cells
        = io.InquireVariable<std::uint64_t>("num_cells");
    adios2::Variable<std::uint32_t> var_num_dofs_per_cell
        = io.InquireVariable<std::uint32_t>("num_dofs_per_cell");

    engine.Get(var_num_nodes, num_nodes_global);
    engine.Get(var_num_cells, num_cells_global);
    engine.Get(var_num_dofs_per_cell, num_dofs_per_cell);

    std::cout << num_nodes_global;
  }

  // Compute local sizes, offsets
  std::array<std::int64_t, 2> _local_range
      = dolfinx::MPI::local_range(rank, num_nodes_global, size);

  std::array<std::uint64_t, 2> local_range{(std::uint64_t)_local_range[0],
                                           (std::uint64_t)_local_range[1]};
  std::uint64_t num_nodes_local = local_range[1] - local_range[0];

  std::array<std::int64_t, 2> _cell_range
      = dolfinx::MPI::local_range(rank, num_cells_global, size);

  std::array<std::uint64_t, 2> cell_range{(std::uint64_t)_cell_range[0],
                                          (std::uint64_t)_cell_range[1]};
  std::uint64_t num_cells_local = cell_range[1] - cell_range[0];

  std::vector<int64_t> input_global_indices(num_nodes_local);
  std::vector<T> x(num_nodes_local * 3);
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
          {{local_range[0]}, {num_nodes_local}});
      engine.Get(var_input_global_indices, input_global_indices.data(),
                 adios2::Mode::Deferred);
    }

    if (var_x)
    {
      var_x.SetSelection({{local_range[0], 0}, {num_nodes_local, 3}});
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

    std::int32_t cell_offset = offsets[0];
    for (auto offset = offsets.begin(); offset != offsets.end(); ++offset)
      *offset -= cell_offset;
  }

  // engine.EndStep();

  std::vector<T> x_reduced(num_nodes_local * dim);
  for (std::uint32_t i = 0; i < num_nodes_local; ++i)
  {
    for (std::uint32_t j = 0; j < (std::uint32_t)dim; ++j)
      x_reduced[i * dim + j] = x[i * 3 + j];
  }

  fem::CoordinateElement<T> element
      = fem::CoordinateElement<T>(cell_type, degree, lagrange_variant);

  std::array<std::size_t, 2> xshape = {num_nodes_local, (std::uint32_t)dim};
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

} // namespace dolfinx::io::native

namespace dolfinx::io::impl_native
{
//-----------------------------------------------------------------------------
std::variant<dolfinx::mesh::Mesh<float>, dolfinx::mesh::Mesh<double>>
read_mesh_variant(adios2::IO& io, adios2::Engine& engine, MPI_Comm comm)
{
  engine.BeginStep();
  std::string floating_point = io.VariableType("x");

  if (floating_point == "float")
  {
    dolfinx::mesh::Mesh<float> mesh
        = dolfinx::io::native::read_mesh<float>(io, engine, comm);
    engine.EndStep();
    return mesh;
  }
  else if (floating_point == "double")
  {
    dolfinx::mesh::Mesh<double> mesh
        = dolfinx::io::native::read_mesh<double>(io, engine, comm);
    engine.EndStep();
    return mesh;
  }
  else
  {
    throw std::runtime_error("Floating point type is neither float or double");
  }
}

} // namespace dolfinx::io::impl_native

#endif
