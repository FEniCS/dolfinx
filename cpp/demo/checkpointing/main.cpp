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
  adios2::ADIOS adios(mesh->comm());
  adios2::IO io = adios.DeclareIO("mesh-write");
  io.SetEngine("BP5");
  adios2::Engine engine = io.Open("mesh.bp", adios2::Mode::Write);

  // TODO: Need to move this inside the checkpointing module
  io.DefineAttribute<std::string>("version", DOLFINX_VERSION_STRING);
  io.DefineAttribute<std::string>("git_hash", DOLFINX_VERSION_GIT);

  io::checkpointing::write_mesh(io, engine, *mesh);

  engine.Close();

  adios2::IO io_read = adios.DeclareIO("mesh-read");
  io_read.SetEngine("BP5");

  adios2::Engine reader = io_read.Open("mesh.bp", adios2::Mode::Read);

  // TODO: move type deduction inside checkpointing
  if ("float" == io.VariableType("x"))
  {
    using T = float;
    mesh::Mesh<T> mesh_read
        = io::checkpointing::read_mesh<T>(io_read, reader, mesh->comm());
    reader.Close();

    adios2::IO io_write = adios.DeclareIO("mesh-rewrite");
    io_write.SetEngine("BP5");

    adios2::Engine writer = io_write.Open("mesh2.bp", adios2::Mode::Write);
    io::checkpointing::write_mesh(io_write, writer, mesh_read);
    writer.Close();
  }
  else if ("double" == io.VariableType("x"))
  {
    using T = double;
    mesh::Mesh<T> mesh_read
        = io::checkpointing::read_mesh<T>(io_read, reader, mesh->comm());
    reader.Close();

    adios2::IO io_write = adios.DeclareIO("mesh-rewrite");
    io_write.SetEngine("BP5");

    adios2::Engine writer = io_write.Open("mesh2.bp", adios2::Mode::Write);
    io::checkpointing::write_mesh(io_write, writer, mesh_read);
    writer.Close();
  }

  MPI_Finalize();
  return 0;
}
