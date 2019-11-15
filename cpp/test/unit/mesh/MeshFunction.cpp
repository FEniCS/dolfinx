// Copyright (C) 2019 Francesco Ballarin
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
//
// Unit tests for Distributed Meshes

#include <catch.hpp>
#include <dolfin.h>

using namespace dolfin;

namespace
{
Eigen::Array<bool, Eigen::Dynamic, 1>
marking_function(const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic, Eigen::RowMajor>>& x)
{
  Eigen::Array<bool, 1, Eigen::Dynamic> inside(x.cols());
  inside.fill(true);
  return inside;
}

void test_mesh_function()
{
  int argc = 0;
  char** argv = nullptr;
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  // Create mesh using all processes
  std::array<Eigen::Vector3d, 2> pt{Eigen::Vector3d(0.0, 0.0, 0.0),
                                    Eigen::Vector3d(1.0, 1.0, 0.0)};
  auto mesh = std::make_shared<mesh::Mesh>(generation::RectangleMesh::create(
      MPI_COMM_WORLD, pt, {{16, 16}}, mesh::CellType::triangle,
      mesh::GhostMode::none));

  // Loop over entity dimension
  for (int d = 0; d <= mesh->geometry().dim(); ++d)
  {
    mesh::MeshFunction<std::size_t> mesh_function(mesh, d, 0.0);
    mesh_function.mark(marking_function, 1.0);
    for (const auto& e : mesh::MeshRange(*mesh, d, mesh::MeshRangeType::ALL))
    {
      CHECK(mesh_function.values()[e.index()] == 1.0);
    }
  }
}
} // namespace

TEST_CASE("Mesh Function", "[mesh_function]")
{
  CHECK_NOTHROW(test_mesh_function());
}
