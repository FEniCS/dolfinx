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
#include <typeinfo>
#include <variant>

using namespace dolfinx;
using namespace dolfinx::io;

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  MPI_Init(&argc, &argv);

  // Create mesh and function space
  auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
  auto mesh = std::make_shared<mesh::Mesh<float>>(mesh::create_rectangle<float>(
      MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}}, {4, 4},
      mesh::CellType::quadrilateral, part));

  try
  {
    // Set up ADIOS2 IO and Engine
    adios2::ADIOS adios(mesh->comm());

    adios2::IO io = adios.DeclareIO("mesh-write");
    io.SetEngine("BP5");
    adios2::Engine engine = io.Open("mesh.bp", adios2::Mode::Append);

    io::native::write_mesh(io, engine, *mesh);
    std::span<float> x = mesh->geometry().x();
    for (std::size_t i = 0; i < x.size(); ++i)
    {
      x[i] *= 4;
    }

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

    // ReadRandomAccess mode : set up ADIOS2 IO and Engine
    adios2::IO io_rra = adios_read.DeclareIO("mesh-rra");
    io_rra.SetEngine("BP5");
    adios2::Engine engine_rra
        = io_rra.Open("mesh.bp", adios2::Mode::ReadRandomAccess);

    // Write mode : set up ADIOS2 IO and Engine
    adios2::IO io_write = adios_read.DeclareIO("mesh-write");
    io_write.SetEngine("BP5");
    adios2::Engine engine_write
        = io_write.Open("mesh2.bp", adios2::Mode::Write);

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
        = io::native::read_mesh<float>(io_read, engine_read, MPI_COMM_WORLD);
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

    io::native::update_mesh<float>(io_read, engine_read, mesh_read, querystep);

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

  MPI_Finalize();
  return 0;
}
