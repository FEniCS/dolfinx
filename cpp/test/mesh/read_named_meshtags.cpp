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
#include <dolfinx/mesh/generation.h>
#include <dolfinx/mesh/utils.h>
#include <numeric>
#include <string>

using namespace dolfinx;

namespace
{
void test_read_named_meshtags()
{
  const std::string mesh_file_name = "Domain.xdmf";
  constexpr std::int32_t domain_value = 1;
  constexpr std::int32_t material_value = 2;

  // Create mesh
  auto part = mesh::create_cell_partitioner(mesh::GhostMode::none);
  auto mesh = std::make_shared<mesh::Mesh<double>>(
      mesh::create_rectangle(MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}}, {3, 3},
                             mesh::CellType::triangle, part));

  const std::int32_t n_cells = mesh->topology()->index_map(2)->size_local();
  std::vector<std::int32_t> indices(n_cells);
  std::iota(std::begin(indices), std::end(indices), 0);

  std::vector<std::int32_t> domain_values(n_cells, domain_value);
  std::vector<std::int32_t> material_values(n_cells, material_value);

  mesh::MeshTags<std::int32_t> mt_domains(mesh->topology(), 2, indices,
                                          domain_values);
  mt_domains.name = "domain";

  mesh::MeshTags<std::int32_t> mt_materials(mesh->topology(), 2, indices,
                                            material_values);
  mt_materials.name = "material";

  io::XDMFFile file(mesh->comm(), mesh_file_name, "w", io::XDMFFile::Encoding::HDF5);
  file.write_mesh(*mesh);
  file.write_meshtags(mt_domains, mesh->geometry(),
                      "/Xdmf/Domain/mesh/Geometry");
  file.write_meshtags(mt_materials, mesh->geometry(),
                      "/Xdmf/Domain/Grid/Geometry");
  file.close();

  io::XDMFFile mesh_file(MPI_COMM_WORLD, mesh_file_name, "r",
                        io::XDMFFile::Encoding::HDF5);
  mesh = std::make_shared<mesh::Mesh<double>>(mesh_file.read_mesh(
      fem::CoordinateElement<double>(mesh::CellType::triangle, 1),
      mesh::GhostMode::none, "mesh"));

  mesh::MeshTags<std::int32_t> mt_first
      = mesh_file.read_meshtags(*mesh, "material", {});
  CHECK(mt_first.values().front() == material_value);

  mesh::MeshTags<std::int32_t> mt_domain
      = mesh_file.read_meshtags(*mesh, "domain", "domain", "/Xdmf/Domain");
  CHECK(mt_domain.values().front() == domain_value);

  mesh::MeshTags<std::int32_t> mt_material
      = mesh_file.read_meshtags(*mesh, "material", "material", "/Xdmf/Domain");
  CHECK(mt_material.values().front() == material_value);

  CHECK_THROWS(mesh_file.read_meshtags(*mesh, "mesh", "missing"));
  mesh_file.close();
}
} // namespace

TEST_CASE("Read meshtag by name", "[read_meshtag_by_name]")
{
  test_read_named_meshtags();
}
