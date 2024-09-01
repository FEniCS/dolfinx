// ```text
// Copyright (C) 2024 Abdullah Mujahid, Jørgen S. Dokken, Jack S. Hale
// This file is part of DOLFINx (https://www.fenicsproject.org)
// SPDX-License-Identifier:    LGPL-3.0-or-later
// ```

// # Checkpointing
//

#include <adios2.h>
#include <dolfinx.h>
#include <dolfinx/common/version.h>
#include <dolfinx/io/ADIOS2_utils.h>
#include <dolfinx/io/checkpointing.h>
#include <mpi.h>
#include <petscsys.h>
#include <typeinfo>
#include <variant>

using namespace dolfinx;
using namespace dolfinx::io;

// Create cell meshtags

template <std::floating_point T>
std::shared_ptr<mesh::MeshTags<std::int32_t>>
create_meshtags(dolfinx::mesh::Mesh<T>& mesh)
{
  // Create cell meshtags
  auto geometry = mesh.geometry();
  auto topology = mesh.topology();

  int dim = geometry.dim();
  topology->create_entities(dim);
  const std::shared_ptr<const dolfinx::common::IndexMap> topo_imap
      = topology->index_map(dim);

  std::int32_t num_entities = topo_imap->size_local();

  auto cmap = geometry.cmap();
  auto geom_layout = cmap.create_dof_layout();
  std::uint32_t num_dofs_per_entity = geom_layout.num_entity_closure_dofs(dim);

  std::vector<int32_t> entities_array(num_entities * num_dofs_per_entity);
  std::vector<int32_t> entities_offsets(num_entities + 1);
  std::uint64_t offset = topo_imap->local_range()[0];
  std::vector<std::int32_t> values(num_entities);

  for (std::int32_t i = 0; i < num_entities; ++i)
  {
    values[i] = i + offset;
  }

  auto entities = topology->connectivity(dim, 0);

  for (int i = 0; i < (int)num_entities + 1; ++i)
    entities_offsets[i] = entities->offsets()[i];

  for (int i = 0; i < (int)(num_entities * num_dofs_per_entity); ++i)
    entities_array[i] = entities->array()[i];

  graph::AdjacencyList<std::int32_t> entities_local(entities_array,
                                                    entities_offsets);

  auto meshtags = std::make_shared<mesh::MeshTags<std::int32_t>>(
      mesh::create_meshtags<std::int32_t>(topology, dim, entities_local,
                                          values));

  return meshtags;
}

template <typename T, typename U>
std::shared_ptr<fem::Function<T>>
create_function(std::shared_ptr<dolfinx::mesh::Mesh<U>> mesh)
{
  auto element = basix::create_element<U>(
      basix::element::family::P, basix::cell::type::triangle, 1,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);

  auto V = std::make_shared<fem::FunctionSpace<U>>(
      fem::create_functionspace(mesh, element, {}));

  // Interpolate sin(2 \pi x[0]) sin(2 \pi x[1]) in the scalar Lagrange finite
  // element space
  auto expression
      = [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
  {
    std::vector<T> f;
    for (std::size_t p = 0; p < x.extent(1); ++p)
    {
      auto x0 = x(0, p);
      auto x1 = x(1, p);
      f.push_back(std::sin(2 * std::numbers::pi * x0)
                  * std::sin(2 * std::numbers::pi * x1));
    }
    return {f, {f.size()}};
  };
  auto f = std::make_shared<fem::Function<T>>(V);
  f->interpolate(expression);

  return f;
}

using T = PetscScalar;
using U = typename dolfinx::scalar_value_type_t<T>;

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  // Create mesh and function space
  auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
  auto mesh = std::make_shared<mesh::Mesh<U>>(
      mesh::create_rectangle<U>(MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}},
                                {4, 4}, mesh::CellType::quadrilateral, part));

  auto meshtags = create_meshtags<U>(*mesh);

  auto f = create_function<T, U>(mesh);

  try
  {
    // Set up ADIOS2 IO and Engine
    adios2::ADIOS adios(mesh->comm());

    {
      adios2::IO io = adios.DeclareIO("mesh-write");
      io.SetEngine("BP5");
      adios2::Engine engine = io.Open("mesh.bp", adios2::Mode::Append);
      io::native::write_mesh(io, engine, *mesh);
      engine.Close();
    }

    {
      adios2::IO io = adios.DeclareIO("meshtags-write");
      io.SetEngine("BP5");
      adios2::Engine engine = io.Open("meshtags.bp", adios2::Mode::Append);

      io::native::write_meshtags<U, std::int32_t>(io, engine, *mesh, *meshtags);
      engine.Close();
    }

    {
      adios2::IO io = adios.DeclareIO("function-write");
      io.SetEngine("BP5");
      adios2::Engine engine = io.Open("function.bp", adios2::Mode::Append);

      io::native::write_function<T, U>(io, engine, *f, *mesh);
      engine.Close();
    }
    adios2::IO io = adios.DeclareIO("mesh-write");
    io.SetEngine("BP5");
    adios2::Engine engine = io.Open("mesh.bp", adios2::Mode::Append);

    io::native::write_mesh(io, engine, *mesh);
    io::native::write_meshtags<float, std::int32_t>(io, engine, *mesh,
                                                    *meshtags);
    std::span<float> x = mesh->geometry().x();
    std::ranges::transform(x, x.begin(), [](auto xi) { return xi *= 4; });

    io::native::write_mesh(io, engine, *mesh, 0.5);

    engine.Close();
  }
  catch (std::exception& e)
  {
    std::cout << "ERROR: ADIOS2 exception: " << e.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  try
  {
    // Read mode : set up ADIOS2 IO and Engine
    adios2::ADIOS adios_read(MPI_COMM_WORLD);
    adios2::IO io_read = adios_read.DeclareIO("mesh-read");
    io_read.SetEngine("BP5");
    adios2::Engine engine_read = io_read.Open("mesh.bp", adios2::Mode::Read);

    engine_read.BeginStep();
    auto mesh_read
        = io::native::read_mesh<float>(io_read, engine_read, MPI_COMM_WORLD);

    mesh::MeshTags<std::int32_t> mt
        = io::native::read_meshtags<float, std::int32_t>(
            io_read, engine_read, mesh_read, "mesh_tags");

    if (engine_read.BetweenStepPairs())
    {
      engine_read.EndStep();
    }

    engine_read.Close();

    // ReadRandomAccess mode : set up ADIOS2 IO and Engine
    adios2::IO io_rra = adios_read.DeclareIO("mesh-rra");
    io_rra.SetEngine("BP5");
    adios2::Engine engine_rra
        = io_rra.Open("mesh.bp", adios2::Mode::ReadRandomAccess);

    {
      adios2::IO io_write = adios_read.DeclareIO("mesh-write");
      io_write.SetEngine("BP5");
      adios2::Engine engine_write
          = io_write.Open("mesh2.bp", adios2::Mode::Write);

      io::native::write_mesh(io_write, engine_write, mesh_read);
      engine_write.Close();
    }
    {
      adios2::IO io_write = adios_read.DeclareIO("meshtags-write");
      io_write.SetEngine("BP5");
      adios2::Engine engine_write
          = io_write.Open("meshtags2.bp", adios2::Mode::Write);

      io::native::write_meshtags<U, std::int32_t>(io_write, engine_write,
                                                  mesh_read, mt);
      engine_write.Close();
    }

    // Find the time stamps array
    auto var_time = io_rra.InquireVariable<double>("time");
    const std::vector<std::vector<adios2::Variable<double>::Info>> timestepsinfo
        = var_time.AllStepsBlocksInfo();

    std::size_t num_steps = timestepsinfo.size();
    std::vector<double> times(num_steps);

    for (std::size_t step = 0; step < num_steps; ++step)
    {
      var_time.SetStepSelection({step, 1});
      engine_rra.Get(var_time, times[step]);
    }
    engine_rra.Close();

    // Read mesh
    engine_read.BeginStep();
    auto mesh_read
        = io::native::read_mesh<U>(io_read, engine_read, MPI_COMM_WORLD);
    if (engine_read.BetweenStepPairs())
    {
      engine_read.EndStep();
    }
    // Write mesh
    io::native::write_mesh(io_write, engine_write, mesh_read);

    // Update mesh
    double time = 0.5;
    std::size_t querystep;
    auto pos = std::ranges::find(times, time);
    if (pos != times.end())
    {
      querystep = std::ranges::distance(times.begin(), pos);
      std::cout << "Query step is : " << querystep << "\n";
    }
    else
    {
      throw std::runtime_error("Step corresponding to time : "
                               + std::to_string(time) + " not found");
    }

    io::native::update_mesh<U>(io_read, engine_read, mesh_read, querystep);

    // Write updated mesh
    io::native::write_mesh(io_write, engine_write, mesh_read, time);

    engine_read.Close();
    engine_write.Close();
  }
  catch (std::exception& e)
  {
    std::cout << "ERROR: ADIOS2 exception: " << e.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  PetscFinalize();
  return 0;
}
