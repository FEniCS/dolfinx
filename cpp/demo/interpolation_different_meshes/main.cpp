// Copyright (C) 2022 Igor A. Baratta and Massimiliano Leoni
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <basix/e-lagrange.h>
#include <dolfinx/common/timing.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/io/ADIOS2Writers.h>
#include <dolfinx/mesh/generation.h>
#include <memory>

using namespace dolfinx;

using T = double;

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  MPI_Init(&argc, &argv);
  {
    MPI_Comm comm{MPI_COMM_WORLD};

    // Create a tetrahedral mesh
    auto mesh_tet = std::make_shared<mesh::Mesh>(
        mesh::create_box(comm, {{{0, 0, 0}, {1, 1, 1}}}, {10, 10, 10},
                         mesh::CellType::tetrahedron, mesh::GhostMode::none));
    // Create a hexahedral mesh
    auto mesh_hex = std::make_shared<mesh::Mesh>(
        mesh::create_box(comm, {{{0, 0, 0}, {1, 1, 1}}}, {9, 8, 7},
                         mesh::CellType::hexahedron, mesh::GhostMode::none));

    basix::FiniteElement eL = basix::element::create_lagrange(
        mesh::cell_type_to_basix_type(mesh_tet->topology().cell_type()), 1,
        basix::element::lagrange_variant::equispaced, false);
    auto V_tet = std::make_shared<fem::FunctionSpace>(
        fem::create_functionspace(mesh_tet, eL, 3));

    basix::FiniteElement eR = basix::element::create_lagrange(
        mesh::cell_type_to_basix_type(mesh_hex->topology().cell_type()), 2,
        basix::element::lagrange_variant::equispaced, false);
    auto V_hex = std::make_shared<fem::FunctionSpace>(
        fem::create_functionspace(mesh_hex, eR, 3));

    auto u_tet = std::make_shared<fem::Function<T>>(V_tet);
    auto u_hex = std::make_shared<fem::Function<T>>(V_hex);

    auto fun = [](auto& x)
    {
      auto r = xt::zeros_like(x);
      xt::row(r, 0) = xt::cos(10 * xt::row(x, 0)) * xt::sin(10 * xt::row(x, 2));
      xt::row(r, 1) = xt::sin(10 * xt::row(x, 0)) * xt::sin(10 * xt::row(x, 2));
      xt::row(r, 2) = xt::cos(10 * xt::row(x, 0)) * xt::cos(10 * xt::row(x, 2));
      return r;
    };

    u_tet->interpolate(fun);
    u_hex->interpolate(*u_tet);

#ifdef HAS_ADIOS2
    io::VTXWriter write_tet(mesh_tet->comm(), "u_tet.vtx", {u_tet});
    write_tet.write(0.0);

    io::VTXWriter write_hex(mesh_hex->comm(), "u_hex.vtx", {u_hex});
    write_hex.write(0.0);
#endif
  }
  MPI_Finalize();

  return 0;
}
