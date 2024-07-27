// ```text
// Copyright (C) 2024 Abdullah Mujahid
// This file is part of DOLFINx (https://www.fenicsproject.org)
// SPDX-License-Identifier:    LGPL-3.0-or-later
// ```

// # Checkpointing
//

#include <adios2.h>
#include <dolfinx.h>
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

  io::checkpointing::write_mesh(io, engine, mesh);

  engine.Close();

  MPI_Finalize();
  return 0;
}
