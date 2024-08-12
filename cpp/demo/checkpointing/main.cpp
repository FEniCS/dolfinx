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

  // {
  //   int i=0;
  //   while (i == 0)
  //     sleep(5);
  // }

  // Create mesh and function space
  auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
  auto mesh = std::make_shared<mesh::Mesh<float>>(mesh::create_rectangle<float>(
      MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}}, {4, 4},
      mesh::CellType::quadrilateral, part));


    try
    {
      // Set up ADIOS2 IO and Engine
      adios2::ADIOS adios("checkpointing.yml", mesh->comm());

      adios2::IO io = adios.DeclareIO("mesh-write");
      io.SetEngine("BP5");
      adios2::Engine engine = io.Open("mesh.bp", adios2::Mode::Write);

      std::vector<std::string> mytags = {"one"};
      io.DefineAttribute<std::string>("tags", mytags.data(), mytags.size(), "", "", true);

      std::vector<std::string> mytags2 = {"one", "two"};
      io.DefineAttribute<std::string>("tags", mytags2.data(), mytags2.size());

      io::native::write_mesh(io, engine, *mesh);

      engine.Close();
    }
    catch (std::exception &e)
    {
        std::cout << "ERROR: ADIOS2 exception: " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

  // Set up ADIOS2 IO and Engine
  // adios2::ADIOS adios_read(mesh->comm());
  // adios2::IO io_read = adios_read.DeclareIO("mesh-read");
  // io_read.SetEngine("BP5");
  // adios2::Engine engine_read = io_read.Open("mesh.bp", adios2::Mode::Read);

  // auto mesh_read_variant
  //     = io::native::read_mesh_variant(io_read, engine_read, MPI_COMM_WORLD);

  // engine_read.Close();

  // ----------------------------------------------------------------------
  // auto adios_query
  //     = ADIOS2Wrapper(MPI_COMM_WORLD, "mesh.bp", "mesh-read", "BP5", "read");

  // auto adios_read
  //     = ADIOS2Wrapper(MPI_COMM_WORLD, "mesh.bp", "mesh-read", "BP5", "read");

  // Can't use the same engine to query as well as read
  // since, in that case BeginStep and EndStep will be called twice
  // on a dataset written with a single Step

  // auto reader = adios_read.engine();
  // reader->BeginStep();
  // auto mesh_read_variant = io::native::read_mesh_variant(
  //     adios_query, adios_read, mesh->comm());

  // reader->EndStep();

  // // We cannot resolve easily the variant
  // // Hence the user can query the type and
  // // call the correct read_mesh
  // auto mesh_read = std::get<0>(mesh_read_variant);
  // // auto mesh_read = std::get<mesh::Mesh<float>>(mesh_read_variant);

  // auto adios_write
  //     = ADIOS2Wrapper(mesh->comm(), "mesh2.bp", "mesh-rewrite", "BP5",
  //     "write");

  // io::native::write_mesh(adios_write, mesh_read);

  // auto io_query = adios_query.io();
  // auto engine_query = adios_query.engine();

  // Following throws an error. engine_read->BeginStep() is needed to
  // read the VariableType, but then EndStep() fails with message
  // EndStep() called without a successful BeginStep()

  // engine_query->BeginStep();
  // std::string floating_point = io_query->VariableType("x");
  // engine_query->EndStep();

  // // std::string floating_point =
  // dolfinx::io::native::query_type(adios_read); std::cout <<
  // floating_point;

  // // TODO: move type deduction inside checkpointing
  // if ("float" == floating_point)
  // {
  //   using T = float;
  //   mesh::Mesh<T> mesh_read
  //       = io::native::read_mesh<T>(adios_read, mesh->comm());
  //   adios_read.close();

  //   auto adios_write = ADIOS2Wrapper(mesh->comm(), "mesh2.bp",
  //   "mesh-rewrite",
  //                                    "BP5", "write");

  //   auto io_write = adios_write.io();

  //   io::native::write_mesh(adios_write, mesh_read);
  //   adios_write.close();
  // }
  // else if ("double" == floating_point)
  // {
  //   using T = double;
  //   mesh::Mesh<T> mesh_read
  //       = io::native::read_mesh<T>(adios_read, mesh->comm());
  //   adios_read.close();

  //   auto adios_write = ADIOS2Wrapper(mesh->comm(), "mesh2.bp",
  //   "mesh-rewrite",
  //                                    "BP5", "write");

  //   auto io_write = adios_write.io();

  //   io::native::write_mesh(adios_write, mesh_read);
  //   adios_write.close();
  // }

  MPI_Finalize();
  return 0;
}
