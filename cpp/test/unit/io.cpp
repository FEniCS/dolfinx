// Copyright (C) 2021 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch.hpp>
#include <dolfinx/generation/RectangleMesh.h>
#include <dolfinx/io/FidesWriter.h>
#include <dolfinx/mesh/Mesh.h>
#include <mpi.h>

using namespace dolfinx;

namespace
{

void test_fides_mesh()
{
  auto mesh = std::make_shared<mesh::Mesh>(generation::RectangleMesh::create(
      MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 0.0}}}, {22, 12},
      mesh::CellType::triangle, mesh::GhostMode::shared_facet));

  io::FidesWriter writer(mesh->mpi_comm(), "test_mesh.bp", mesh);
  writer.write(0.0);
}

} // namespace

TEST_CASE("Fides mesh output", "[fides_mesh_write]")
{
  CHECK_NOTHROW(test_fides_mesh());
}
