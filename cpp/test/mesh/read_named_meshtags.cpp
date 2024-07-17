// Copyright (C) 2024 Massimiliano Leoni
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
//
// Unit tests for reading meshtags by name

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/la/Vector.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/mesh/utils.h>

#include <string>

using namespace dolfinx;

namespace
{

void test_read_named_meshtags()
{
  const std::string mesh_file = "Domain.xdmf";

  io::XDMFFile meshFile(MPI_COMM_WORLD, mesh_file, "r",
                        io::XDMFFile::Encoding::ASCII);
  auto mesh = std::make_shared<mesh::Mesh<double>>(meshFile.read_mesh(
      fem::CoordinateElement<double>(mesh::CellType::tetrahedron, 1),
      mesh::GhostMode::none, "Grid"));

  const auto mt_domain = meshFile.read_meshtags_by_label(
      *mesh, "Grid", "domain", "/Xdmf/Domain");

  CHECK(mt_domain.values().front() == 0);

  const auto mt_material = meshFile.read_meshtags_by_label(
      *mesh, "Grid", "material", "/Xdmf/Domain");

  CHECK(mt_material.values().front() == 1);

  CHECK_THROWS(meshFile.read_meshtags_by_label(*mesh, "Grid", "missing"));
}

} // namespace

TEST_CASE("Read meshtag by name", "[read_meshtag_by_name]")
{
  CHECK_NOTHROW(test_read_named_meshtags());
}
