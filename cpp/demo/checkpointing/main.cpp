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
    adios2::Engine engine = io.Open("mesh.bp", adios2::Mode::Write);

    io::native::write_mesh(io, engine, *mesh);

    engine.Close();
  }
  catch (std::exception& e)
  {
    std::cout << "ERROR: ADIOS2 exception: " << e.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  try
  {
    // Set up ADIOS2 IO and Engine
    adios2::ADIOS adios_read(MPI_COMM_WORLD);
    adios2::IO io_read = adios_read.DeclareIO("mesh-read");
    io_read.SetEngine("BP5");
    adios2::Engine engine_read = io_read.Open("mesh.bp", adios2::Mode::Read);

    engine_read.BeginStep();
    auto mesh_read
        = io::native::read_mesh<float>(io_read, engine_read, MPI_COMM_WORLD);
    if (engine_read.BetweenStepPairs())
    {
      engine_read.EndStep();
    }

    engine_read.Close();

    adios2::IO io_write = adios_read.DeclareIO("mesh-write");
    io_write.SetEngine("BP5");
    adios2::Engine engine_write
        = io_write.Open("mesh2.bp", adios2::Mode::Write);

    io::native::write_mesh(io_write, engine_write, mesh_read);
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
