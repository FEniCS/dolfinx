// Copyright (C) 2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstdint>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx/mesh/permutationcomputation.h>
#include <mpi.h>
#include <vector>

using namespace dolfinx;

namespace
{
/// Whether any rank holds a non-zero permutation.
bool any_nonzero(const std::vector<std::uint8_t>& perms)
{
  int local = std::ranges::any_of(perms, [](std::uint8_t p) { return p != 0; });
  MPI_Allreduce(MPI_IN_PLACE, &local, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
  return local;
}

/// Check, for every dimension with an orientation, that the
/// permutations Topology unpacks from the packed cell permutations
/// match those measured from the mesh entities
void check_mesh(const mesh::Mesh<double>& mesh)
{
  mesh::Topology& topology = *mesh.topology_mutable();
  for (int dim = 1; dim < topology.dim(); ++dim)
  {
    topology.create_entities(dim);
    topology.create_entity_permutations(dim);
    const std::vector<std::uint8_t>& perms
        = topology.get_entity_permutations(dim);
    CHECK(perms == mesh::compute_entity_permutations(topology, dim, 1));
    CHECK(any_nonzero(perms));
  }
}
} // namespace

TEST_CASE("Entity permutations match the packed cell permutations",
          "[mesh][permutations]")
{
  constexpr std::array<std::array<double, 2>, 2> rectangle = {{{0, 0}, {1, 1}}};
  constexpr std::array<std::array<double, 3>, 2> box = {{{0, 0, 0}, {1, 1, 1}}};

  const mesh::CellType cell_type
      = GENERATE(mesh::CellType::triangle, mesh::CellType::quadrilateral,
                 mesh::CellType::tetrahedron, mesh::CellType::hexahedron,
                 mesh::CellType::prism);
  CAPTURE(cell_type);
  if (mesh::cell_dim(cell_type) == 2)
  {
    check_mesh(mesh::create_rectangle<double>(MPI_COMM_WORLD, rectangle, {3, 3},
                                              cell_type));
  }
  else
  {
    check_mesh(
        mesh::create_box<double>(MPI_COMM_WORLD, box, {2, 2, 2}, cell_type));
  }
}
