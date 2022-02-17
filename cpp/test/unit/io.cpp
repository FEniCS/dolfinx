// Copyright (C) 2021 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_ADIOS2

#include <catch2/catch.hpp>
#include <dolfinx/io/ADIOS2Writers.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/generation.h>
#include <mpi.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

using namespace dolfinx;

namespace
{

void test_fides_mesh()
{
  auto mesh = std::make_shared<mesh::Mesh>(mesh::create_rectangle(
      MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}}, {22, 12},
      mesh::CellType::triangle, mesh::GhostMode::shared_facet));
  io::FidesWriter writer(mesh->comm(), "test_mesh.bp", mesh);
  writer.write(0.0);
  auto points
      = xt::adapt(mesh->geometry().x().data(), mesh->geometry().x().size(),
                  xt::no_ownership(),
                  std::vector{mesh->geometry().x().size() / 3, std::size_t(3)});
  // Move all coordinates of the mesh geometry
  points += 1;
  writer.write(0.2);

  // Only move x coordinate
  auto x_coords = xt::view(points, xt::all(), 0);
  x_coords -= 0.5;
  writer.write(0.4);
}

} // namespace

TEST_CASE("Fides mesh output", "[fides_mesh_write]")
{
  CHECK_NOTHROW(test_fides_mesh());
}

#endif
