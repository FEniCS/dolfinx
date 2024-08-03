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

  // Set up ADIOS2 IO and Engine
  auto adios
      = ADIOS2Wrapper(mesh->comm(), "mesh.bp", "mesh-write", "BP5", "write");

  auto io = adios.io();

  // TODO: Need to move this inside the checkpointing module
  io->DefineAttribute<std::string>("version", DOLFINX_VERSION_STRING);
  io->DefineAttribute<std::string>("git_hash", DOLFINX_VERSION_GIT);

  // io::checkpointing::write_mesh(io, engine, *mesh);
  io::checkpointing::write_mesh(adios, *mesh);

  adios.close();

  // ----------------------------------------------------------------------
  auto adios_query
      = ADIOS2Wrapper(mesh->comm(), "mesh.bp", "mesh-read", "BP5", "read");

  auto adios_read
      = ADIOS2Wrapper(mesh->comm(), "mesh.bp", "mesh-read", "BP5", "read");

  // Can't use the same engine to query as well as read
  // since, in that case BeginStep and EndStep will be called twice
  // on a dataset written with a single Step
  auto mesh_read_variant = io::checkpointing::read_mesh_variant(
      adios_query, adios_read, mesh->comm());

  // We cannot resolve easily the variant
  // Hence the user can query the type and
  // call the correct read_mesh
  auto mesh_read = std::get<0>(mesh_read_variant);
  // auto mesh_read = std::get<mesh::Mesh<float>>(mesh_read_variant);

  auto adios_write
      = ADIOS2Wrapper(mesh->comm(), "mesh2.bp", "mesh-rewrite", "BP5", "write");

  io::checkpointing::write_mesh(adios_write, mesh_read);

  // auto io_query = adios_query.io();
  // auto engine_query = adios_query.engine();

  // Following throws an error. engine_read->BeginStep() is needed to
  // read the VariableType, but then EndStep() fails with message
  // EndStep() called without a successful BeginStep()

  // engine_query->BeginStep();
  // std::string floating_point = io_query->VariableType("x");
  // engine_query->EndStep();

  // // std::string floating_point =
  // dolfinx::io::checkpointing::query_type(adios_read); std::cout <<
  // floating_point;

  // // TODO: move type deduction inside checkpointing
  // if ("float" == floating_point)
  // {
  //   using T = float;
  //   mesh::Mesh<T> mesh_read
  //       = io::checkpointing::read_mesh<T>(adios_read, mesh->comm());
  //   adios_read.close();

  //   auto adios_write = ADIOS2Wrapper(mesh->comm(), "mesh2.bp",
  //   "mesh-rewrite",
  //                                    "BP5", "write");

  //   auto io_write = adios_write.io();

  //   io::checkpointing::write_mesh(adios_write, mesh_read);
  //   adios_write.close();
  // }
  // else if ("double" == floating_point)
  // {
  //   using T = double;
  //   mesh::Mesh<T> mesh_read
  //       = io::checkpointing::read_mesh<T>(adios_read, mesh->comm());
  //   adios_read.close();

  //   auto adios_write = ADIOS2Wrapper(mesh->comm(), "mesh2.bp",
  //   "mesh-rewrite",
  //                                    "BP5", "write");

  //   auto io_write = adios_write.io();

  //   io::checkpointing::write_mesh(adios_write, mesh_read);
  //   adios_write.close();
  // }

  MPI_Finalize();
  return 0;
}
