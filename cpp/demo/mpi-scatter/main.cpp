// Copyright (C) 2022 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <basix/e-lagrange.h>
#include <basix/e-nedelec.h>
#include <cmath>
#include <dolfinx/common/Scatterer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/io/ADIOS2Writers.h>
#include <dolfinx/io/VTKFile.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/generation.h>
#include <filesystem>
#include <mpi.h>

template <typename Vector>
void debug_vector(const Vector& vec)
{
  for (int i = 0; i < dolfinx::MPI::size(MPI_COMM_WORLD); i++)
  {
    MPI_Barrier(MPI_COMM_WORLD);
    if (int rank = dolfinx::MPI::rank(MPI_COMM_WORLD); rank == i)
    {
      std::cout << "\n Rank " << i << std::endl;
      for (auto e : vec)
        std::cout << e << " ";
      std::cout << std::endl;
    }
  }
}

using namespace dolfinx;

/// This program shows how to create finite element spaces without FFCx
/// generated code
int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  MPI_Init(&argc, &argv);

  // The main body of the function is scoped with the curly braces to
  // ensure that all objects that depend on an MPI communicator are
  // destroyed before MPI is finalised at the end of this function.
  {
    MPI_Comm comm = MPI_COMM_WORLD;
    // Create a mesh. For what comes later in this demo we need to
    // ensure that a boundary between cells is located at x0=0.5
    auto mesh = std::make_shared<mesh::Mesh>(mesh::create_rectangle(
        comm, {{{0.0, 0.0}, {1.0, 1.0}}}, {5, 5}, mesh::CellType::triangle,
        mesh::GhostMode::none));

    // Interpolate a function in a scalar Lagrange space and output the
    // result to file for visualisation
    // Create a Basix continuous Lagrange element of degree 1
    basix::FiniteElement e = basix::element::create_lagrange(
        mesh::cell_type_to_basix_type(mesh::CellType::triangle), 1,
        basix::element::lagrange_variant::equispaced, false);

    // Create a scalar function space
    auto V = std::make_shared<fem::FunctionSpace>(
        fem::create_functionspace(mesh, e, 1));

    // Create a finite element Function
    auto u = std::make_shared<fem::Function<double>>(V);

    auto vector = u->x()->mutable_array();

    std::shared_ptr<const common::IndexMap> map = V->dofmap()->index_map;
    int bs = V->dofmap()->index_map_bs();

    common::Scatterer sct(map, bs);
    la::Vector<double> vec(map, bs);

    std::int32_t n = map->size_local() * bs;
    // std::int64_t offset = map->local_range()[0];

    // {
    //   auto x = vec.mutable_array();
    //   std::iota(x.begin(), x.end(), 0);
    //   std::for_each(x.begin(), x.end(), [offset](auto& e) { e += offset; });
    //   xtl::span<const double> x_local = x.subspan(0, n);
    //   xtl::span<double> x_remote = x.subspan(n, map->num_ghosts() * bs);
    //   std::fill(x_remote.begin(), x_remote.end(), 0);
    //   sct.scatter_fwd<double>(x_local, x_remote, common::Scatterer::gather(),
    //                           common::Scatterer::scatter());
    // }
    {
      auto x = vec.mutable_array();
      std::fill(x.begin(), x.end(), 0);
      std::fill(x.begin() + n, x.end(), dolfinx::MPI::rank(comm) + 1);
      xtl::span<double> x_local = x.subspan(0, n);
      xtl::span<const double> x_remote = x.subspan(n, map->num_ghosts() * bs);

      sct.scatter_rev<double>(x_local, x_remote, std::plus<double>(),
                              common::Scatterer::pack(),
                              common::Scatterer::unpack());

      // debug_vector(x_local);
    }
  }

  MPI_Finalize();

  return 0;
}
