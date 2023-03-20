// Copyright (C) 2021 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_ADIOS2

#include <algorithm>
#include <catch2/catch.hpp>
#include <dolfinx/io/ADIOS2Writers.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/generation.h>
#include <mpi.h>

using namespace dolfinx;

namespace
{

void test_fides_mesh()
{
  auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
  auto mesh = std::make_shared<mesh::Mesh<double>>(
      mesh::create_rectangle<double>(MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}},
                                     {22, 12}, mesh::CellType::triangle, part));
  io::FidesWriter writer(mesh->comm(), "test_mesh.bp", mesh);
  writer.write(0.0);

  auto x = mesh->geometry().x();

  // Move all coordinates of the mesh geometry
  std::transform(x.begin(), x.end(), x.begin(), [](auto x) { return x + 1; });
  writer.write(0.2);

  // Only move x coordinate
  for (std::size_t i = 0; i < x.size(); i += 3)
    x[i] -= 0.5;

  writer.write(0.4);
}

} // namespace

TEST_CASE("Fides mesh output", "[fides_mesh_write]")
{
  CHECK_NOTHROW(test_fides_mesh());
}

#endif
