// ```text
// Copyright (C) 2024 Abdullah Mujahid
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

  // Set up ADIOS2 IO and Engine
  auto adios
      = ADIOS2Wrapper(mesh->comm(), "mesh.bp", "mesh-write", "BP5", "write");

  auto io = adios.io();

  // TODO: Need to move this inside the checkpointing module
  io->DefineAttribute<std::string>("version", DOLFINX_VERSION_STRING);
  io->DefineAttribute<std::string>("git_hash", DOLFINX_VERSION_GIT);

  // io::checkpointing::write_mesh(io, engine, *mesh);
  io::checkpointing::write_mesh(adios, *mesh);

  // adios.close();

  // ----------------------------------------------------------------------
  auto adios_read
      = ADIOS2Wrapper(mesh->comm(), "mesh.bp", "mesh-read", "BP5", "read");

  auto io_read = adios_read.io();
  auto engine_read = adios_read.engine();

  // Following throws an error. engine_read->BeginStep() is needed to
  // read the VariableType, but then EndStep() fails with message
  // EndStep() called without a successful BeginStep()

  // engine_read->BeginStep();
  std::string floating_point = io->VariableType("x");
  // engine_read->EndStep();

  // TODO: move type deduction inside checkpointing
  if ("float" == floating_point)
  {
    using T = float;
    mesh::Mesh<T> mesh_read
        = io::checkpointing::read_mesh<T>(adios_read, mesh->comm());
    adios_read.close();

    auto adios_write = ADIOS2Wrapper(mesh->comm(), "mesh2.bp", "mesh-rewrite",
                                     "BP5", "write");

    auto io_write = adios_write.io();

    io::checkpointing::write_mesh(adios_write, mesh_read);
    adios_write.close();
  }
  else if ("double" == floating_point)
  {
    using T = double;
    mesh::Mesh<T> mesh_read
        = io::checkpointing::read_mesh<T>(adios_read, mesh->comm());
    adios_read.close();

    auto adios_write = ADIOS2Wrapper(mesh->comm(), "mesh2.bp", "mesh-rewrite",
                                     "BP5", "write");

    auto io_write = adios_write.io();

    io::checkpointing::write_mesh(adios_write, mesh_read);
    adios_write.close();
  }

  auto container
      = ADIOS2Wrapper(mesh->comm(), "test.bp", "test-write", "BP5", "write");

  io::checkpointing::write_test(container);

  MPI_Finalize();
  return 0;
}
