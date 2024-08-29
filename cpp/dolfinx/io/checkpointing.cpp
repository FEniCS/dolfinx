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
/// basix enum to string
std::map<basix::element::lagrange_variant, std::string> variant_to_string{
    {basix::element::lagrange_variant::unset, "unset"},
    {basix::element::lagrange_variant::equispaced, "equispaced"},
    {basix::element::lagrange_variant::gll_warped, "gll_warped"},
    {basix::element::lagrange_variant::gll_isaac, "gll_isaac"},
    {basix::element::lagrange_variant::gll_centroid, "gll_centroid"},
    {basix::element::lagrange_variant::chebyshev_warped, "chebyshev_warped"},
    {basix::element::lagrange_variant::chebyshev_isaac, "chebyshev_isaac"},
    {basix::element::lagrange_variant::gl_warped, "gl_warped"},
    {basix::element::lagrange_variant::gl_isaac, "gl_isaac"},
    {basix::element::lagrange_variant::gl_centroid, "gl_centroid"},
    {basix::element::lagrange_variant::legendre, "legendre"},
    {basix::element::lagrange_variant::bernstein, "bernstein"},
};

/// string to basix enum
std::map<std::string, basix::element::lagrange_variant> string_to_variant{
    {"unset", basix::element::lagrange_variant::unset},
    {"equispaced", basix::element::lagrange_variant::equispaced},
    {"gll_warped", basix::element::lagrange_variant::gll_warped},
    {"gll_isaac", basix::element::lagrange_variant::gll_isaac},
    {"gll_centroid", basix::element::lagrange_variant::gll_centroid},
    {"chebyshev_warped", basix::element::lagrange_variant::chebyshev_warped},
    {"chebyshev_isaac", basix::element::lagrange_variant::chebyshev_isaac},
    {"gl_warped", basix::element::lagrange_variant::gl_warped},
    {"gl_isaac", basix::element::lagrange_variant::gl_isaac},
    {"gl_centroid", basix::element::lagrange_variant::gl_centroid},
    {"legendre", basix::element::lagrange_variant::legendre},
    {"bernstein", basix::element::lagrange_variant::bernstein},
};

template <class T>
adios2::Variable<T> define_var(adios2::IO& io, std::string name,
                               const adios2::Dims& shape = adios2::Dims(),
                               const adios2::Dims& start = adios2::Dims(),
                               const adios2::Dims& count = adios2::Dims())
{
  if (adios2::Variable var = io.InquireVariable<T>(name); var)
  {
    if (var.Count() != count)
      var.SetSelection({start, count});

    return var;
  }
  else
    return io.DefineVariable<T>(name, shape, start, count,
                                adios2::ConstantDims);
}

} // namespace

namespace dolfinx::io::native
{
//-----------------------------------------------------------------------------
template <std::floating_point T>
void write_mesh(adios2::IO& io, adios2::Engine& engine,
                const dolfinx::mesh::Mesh<T>& mesh, double time)
{

  const mesh::Geometry<T>& geometry = mesh.geometry();
  std::shared_ptr<const mesh::Topology> topology = mesh.topology();
  assert(topology);
  std::shared_ptr<const common::IndexMap> geom_imap = geometry.index_map();

  std::size_t currentstep = engine.CurrentStep();
  assert(!engine.BetweenStepPairs());
  engine.BeginStep();

  if (currentstep == 0)
  {
    // Variables/attributes to save
    std::int32_t dim = geometry.dim();
    std::int32_t tdim = topology->dim();
    std::uint64_t num_cells_global, cell_offset;
    std::uint32_t num_cells_local, degree;
    std::string cell_type, lagrange_variant;
    std::vector<std::int64_t> array_global;
    std::vector<std::int32_t> offsets_global;

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

    std::uint32_t num_dofs_per_cell;
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

      // FIXME: use std::ranges::transform
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

      adios2::Variable<std::uint64_t> var_num_cells
          = io.DefineVariable<std::uint64_t>("num_cells");

      adios2::Variable<std::int64_t> var_topology_array
          = io.DefineVariable<std::int64_t>(
              "topology_array", {num_cells_global * num_dofs_per_cell},
              {cell_offset * num_dofs_per_cell},
              {num_cells_local * num_dofs_per_cell}, adios2::ConstantDims);

      adios2::Variable<std::int32_t> var_topology_offsets
          = io.DefineVariable<std::int32_t>(
              "topology_offsets", {num_cells_global + 1}, {cell_offset},
              {num_cells_local + 1}, adios2::ConstantDims);

      engine.Put(var_num_cells, num_cells_global);
      engine.Put(var_topology_array, array_global.data());
      engine.Put(var_topology_offsets, offsets_global.data());
    }
  }

  std::uint64_t num_nodes_global, offset;
  std::uint32_t num_nodes_local;
  // Nodes information
  {
    // geom_imap = geometry.index_map();
    num_nodes_global = geom_imap->size_global();
    num_nodes_local = geom_imap->size_local();
    offset = geom_imap->local_range()[0];
  }
  const std::span<const T> mesh_x = geometry.x();

  if (currentstep == 0)
  {
    adios2::Variable<std::uint64_t> var_num_nodes
        = io.DefineVariable<std::uint64_t>("num_nodes");
    engine.Put(var_num_nodes, num_nodes_global);
  }

  adios2::Variable var_time = define_var<double>(io, "time", {}, {}, {});

  adios2::Variable var_x = define_var<T>(io, "x", {num_nodes_global, 3},
                                         {offset, 0}, {num_nodes_local, 3});

  engine.Put(var_time, time);
  engine.Put(var_x, mesh_x.subspan(0, num_nodes_local * 3).data());

  engine.EndStep();

  spdlog::info("Mesh written");
}

//-----------------------------------------------------------------------------
/// @cond
template void write_mesh<float>(adios2::IO& io, adios2::Engine& engine,
                                const dolfinx::mesh::Mesh<float>& mesh,
                                double time);

template void write_mesh<double>(adios2::IO& io, adios2::Engine& engine,
                                 const dolfinx::mesh::Mesh<double>& mesh,
                                 double time);

/// @endcond

//-----------------------------------------------------------------------------
template <std::floating_point T>
dolfinx::mesh::Mesh<T> read_mesh(adios2::IO& io, adios2::Engine& engine,
                                 MPI_Comm comm,
                                 dolfinx::mesh::GhostMode ghost_mode)
{

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (!engine.BetweenStepPairs())
  {
    engine.BeginStep();
  }

  // Compatibility check for version and git commit hash
  {
    adios2::Attribute<std::string> var_version
        = io.InquireAttribute<std::string>("version");
    adios2::Attribute<std::string> var_hash
        = io.InquireAttribute<std::string>("git_hash");
    std::string version = var_version.Data()[0];
    if (version != dolfinx::version())
    {
      throw std::runtime_error("Reading from version: " + dolfinx::version()
                               + " written with version: " + version);
    }

    std::string git_hash = var_hash.Data()[0];
    if (git_hash != dolfinx::git_commit_hash())
    {
      throw std::runtime_error("Reading from GIT_COMMIT_HASH: "
                               + dolfinx::git_commit_hash()
                               + " written with GIT_COMMIT_HASH: " + git_hash);
    }
  }
  // Attributes
  std::string name;
  std::int32_t dim, tdim, degree;
  mesh::CellType cell_type;
  basix::element::lagrange_variant lagrange_variant;

  // Read attributes
  {
    adios2::Attribute<std::string> var_name
        = io.InquireAttribute<std::string>("name");
    adios2::Attribute<std::int32_t> var_dim
        = io.InquireAttribute<std::int32_t>("dim");
    adios2::Attribute<std::int32_t> var_tdim
        = io.InquireAttribute<std::int32_t>("tdim");
    adios2::Attribute<std::string> var_cell_type
        = io.InquireAttribute<std::string>("cell_type");
    adios2::Attribute<std::int32_t> var_degree
        = io.InquireAttribute<std::int32_t>("degree");
    adios2::Attribute<std::string> var_variant
        = io.InquireAttribute<std::string>("lagrange_variant");

    name = var_name.Data()[0];
    dim = var_dim.Data()[0];
    tdim = var_tdim.Data()[0];
    cell_type = mesh::to_type(var_cell_type.Data()[0]);
    degree = var_degree.Data()[0];
    lagrange_variant = string_to_variant[var_variant.Data()[0]];
  }

  spdlog::info(
      "Reading mesh with geometric dimension: {} and topological dimension: {}",
      dim, tdim);

  // Scalar variables
  std::uint64_t num_nodes_global, num_cells_global;

  // Read scalar variables
  {
    adios2::Variable<std::uint64_t> var_num_nodes
        = io.InquireVariable<std::uint64_t>("num_nodes");
    adios2::Variable<std::uint64_t> var_num_cells
        = io.InquireVariable<std::uint64_t>("num_cells");

    engine.Get(var_num_nodes, num_nodes_global);
    engine.Get(var_num_cells, num_cells_global);
  }

  auto [x_reduced, x_shape] = dolfinx::io::impl_native::read_geometry_data<T>(
      io, engine, dim, num_nodes_global, rank, size);

  std::vector<int64_t> array = dolfinx::io::impl_native::read_topology_data(
      io, engine, num_cells_global, rank, size);

  assert(engine.BetweenStepPairs());
  engine.EndStep();

  fem::CoordinateElement<T> element
      = fem::CoordinateElement<T>(cell_type, degree, lagrange_variant);

  auto part = mesh::create_cell_partitioner(ghost_mode);

  if (size == 1)
    part = nullptr;

  mesh::Mesh<T> mesh = mesh::create_mesh(comm, comm, array, element, comm,
                                         x_reduced, x_shape, part);

  mesh.name = name;

  return mesh;
}

//-----------------------------------------------------------------------------
/// @cond
template dolfinx::mesh::Mesh<float>
read_mesh<float>(adios2::IO& io, adios2::Engine& engine, MPI_Comm comm,
                 dolfinx::mesh::GhostMode ghost_mode);

template dolfinx::mesh::Mesh<double>
read_mesh<double>(adios2::IO& io, adios2::Engine& engine, MPI_Comm comm,
                  dolfinx::mesh::GhostMode ghost_mode);

/// @endcond

//-----------------------------------------------------------------------------
template <std::floating_point T>
void update_mesh(adios2::IO& io, adios2::Engine& engine,
                 dolfinx::mesh::Mesh<T>& mesh, std::size_t step)
{
  if (!engine.BetweenStepPairs())
  {
    engine.BeginStep();
  }

  std::uint32_t num_nodes_local = mesh.geometry().index_map()->size_local();
  std::vector<T> x_raw(num_nodes_local * 3);

  // Read variables
  {
    double time;
    adios2::Variable<double> var_time = io.InquireVariable<double>("time");
    var_time.SetStepSelection({step, 1});
    if (var_time)
    {
      engine.Get(var_time, time);
      spdlog::info("Updating geometry at time : {}", time);
    }
    else
    {
      throw std::runtime_error("Step : " + std::to_string(step) + " not found");
    }

    adios2::Variable<T> var_x = io.InquireVariable<T>("x");
    var_x.SetStepSelection({step, 1});
    if (var_x)
    {
      std::uint64_t nodes_offset
          = mesh.geometry().index_map()->local_range()[0];
      var_x.SetSelection({{nodes_offset, 0}, {num_nodes_local, 3}});
      engine.Get(var_x, x_raw.data(), adios2::Mode::Sync);
    }
    else
    {
      throw std::runtime_error("Coordinates data not found at step : " + step);
    }
  }

  engine.EndStep();

  // Redistribute adios2 input coordinate data and find updated coordinates of
  // the mesh
  std::vector<T> x_new = dolfinx::MPI::distribute_data(
      mesh.comm(), mesh.geometry().input_global_indices(), mesh.comm(), x_raw,
      3);

  std::span<T> x = mesh.geometry().x();
  for (std::size_t i = 0; i < num_nodes_local * 3; ++i)
  {
    x[i] = x_new[i];
  }
}

//-----------------------------------------------------------------------------
/// @cond
template void update_mesh<float>(adios2::IO& io, adios2::Engine& engine,
                                 dolfinx::mesh::Mesh<float>& mesh,
                                 std::size_t step);

template void update_mesh<double>(adios2::IO& io, adios2::Engine& engine,
                                  dolfinx::mesh::Mesh<double>& mesh,
                                  std::size_t step);

/// @endcond

//-----------------------------------------------------------------------------
std::vector<double> read_timestamps(adios2::IO& io, adios2::Engine& engine)
{
  if (engine.OpenMode() != adios2::Mode::ReadRandomAccess)
  {
    throw std::runtime_error(
        "Time stamps can only be read in ReadRandomAccess mode");
  }
  adios2::Variable<double> var_time = io.InquireVariable<double>("time");
  const std::vector<std::vector<adios2::Variable<double>::Info>> timestepsinfo
      = var_time.AllStepsBlocksInfo();

  std::size_t num_steps = timestepsinfo.size();
  std::vector<double> times(num_steps);

  for (std::size_t step = 0; step < num_steps; ++step)
  {
    var_time.SetStepSelection({step, 1});
    engine.Get(var_time, times[step]);
  }
  return times;
}

} // namespace dolfinx::io::native

namespace dolfinx::io::impl_native
{
//-----------------------------------------------------------------------------
std::pair<std::uint64_t, std::uint64_t> get_counters(int rank, std::uint64_t N,
                                                     int size)
{
  assert(rank >= 0);
  assert(size > 0);

  // Compute number of items per rank and remainder
  const std::uint64_t n = N / size;
  const std::uint64_t r = N % size;

  // Compute local range
  if (static_cast<std::uint64_t>(rank) < r)
    return {rank * (n + 1), n + 1};
  else
    return {rank * n + r, n};
}

//-----------------------------------------------------------------------------
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
read_geometry_data(adios2::IO& io, adios2::Engine& engine, int dim,
                   std::uint64_t num_nodes_global, int rank, int size)
{
  auto [nodes_offset, num_nodes_local]
      = dolfinx::io::impl_native::get_counters(rank, num_nodes_global, size);
  std::vector<T> x(num_nodes_local * 3);
  adios2::Variable<T> var_x = io.InquireVariable<T>("x");
  if (var_x)
  {
    var_x.SetSelection({{nodes_offset, 0}, {num_nodes_local, 3}});
    engine.Get(var_x, x.data(), adios2::Mode::Sync);
  }
  else
  {
    throw std::runtime_error("Coordinates data not found");
  }

  // FIXME: Use std::ranges::transform
  std::vector<T> x_reduced(num_nodes_local * dim);
  for (std::uint32_t i = 0; i < num_nodes_local; ++i)
  {
    for (std::uint32_t j = 0; j < (std::uint32_t)dim; ++j)
      x_reduced[i * dim + j] = x[i * 3 + j];
  }

  std::array<std::size_t, 2> xshape = {num_nodes_local, (std::uint32_t)dim};

  return {std::move(x_reduced), xshape};
}

//-----------------------------------------------------------------------------
std::vector<int64_t> read_topology_data(adios2::IO& io, adios2::Engine& engine,
                                        std::uint64_t num_cells_global,
                                        int rank, int size)
{
  auto [cells_offset, num_cells_local]
      = dolfinx::io::impl_native::get_counters(rank, num_cells_global, size);
  std::vector<int32_t> offsets(num_cells_local + 1);

  adios2::Variable<std::int64_t> var_topology_array
      = io.InquireVariable<std::int64_t>("topology_array");

  adios2::Variable<std::int32_t> var_topology_offsets
      = io.InquireVariable<std::int32_t>("topology_offsets");

  if (var_topology_offsets)
  {
    var_topology_offsets.SetSelection({{cells_offset}, {num_cells_local + 1}});
    engine.Get(var_topology_offsets, offsets.data(), adios2::Mode::Sync);
  }
  else
  {
    throw std::runtime_error("Topology offsets not found");
  }

  std::uint64_t count
      = static_cast<std::uint64_t>(offsets[num_cells_local] - offsets[0]);
  std::vector<int64_t> array(count);

  if (var_topology_array)
  {
    var_topology_array.SetSelection(
        {{static_cast<std::uint64_t>(offsets[0])}, {count}});
    engine.Get(var_topology_array, array.data(), adios2::Mode::Sync);
  }
  else
  {
    throw std::runtime_error("Topology array not found");
  }

  return array;
}

//-----------------------------------------------------------------------------
std::variant<dolfinx::mesh::Mesh<float>, dolfinx::mesh::Mesh<double>>
read_mesh_variant(adios2::IO& io, adios2::Engine& engine, MPI_Comm comm,
                  dolfinx::mesh::GhostMode ghost_mode)
{
  engine.BeginStep();
  std::string floating_point = io.VariableType("x");

  if (floating_point == "float")
  {
    dolfinx::mesh::Mesh<float> mesh
        = dolfinx::io::native::read_mesh<float>(io, engine, comm, ghost_mode);
    if (engine.BetweenStepPairs())
    {
      engine.EndStep();
    }
    return mesh;
  }
  else if (floating_point == "double")
  {
    dolfinx::mesh::Mesh<double> mesh
        = dolfinx::io::native::read_mesh<double>(io, engine, comm, ghost_mode);
    if (engine.BetweenStepPairs())
    {
      engine.EndStep();
    }
    return mesh;
  }
  else
  {
    throw std::runtime_error("Floating point type is neither float nor double");
  }
}

} // namespace dolfinx::io::impl_native

#endif
