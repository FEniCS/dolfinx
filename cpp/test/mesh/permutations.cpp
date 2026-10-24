// Copyright (C) 2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
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
/// Whether any rank holds a non-zero permutation. The meshes below are
/// built so that some entity really is permuted, without which the
/// comparisons would hold vacuously.
bool any_nonzero(const std::vector<std::uint8_t>& perms)
{
  int local = std::ranges::any_of(perms, [](std::uint8_t p) { return p != 0; });
  MPI_Allreduce(MPI_IN_PLACE, &local, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
  return local;
}

/// Check that the permutations of the dimension `dim` entities are the
/// bits the packed cell permutations devote to them: three per face,
/// starting at bit 0, followed by one per edge.
void check_dimension(mesh::Topology& topology, int dim)
{
  const int tdim = topology.dim();
  topology.create_entities(dim);
  const std::vector<std::uint8_t> perms
      = mesh::compute_entity_permutations(topology, dim, 1);
  const std::vector<std::uint32_t> info
      = mesh::compute_cell_permutations(topology, 1);

  const int num_entities = mesh::cell_num_entities(topology.cell_type(), dim);
  const int bits = dim == 2 ? 3 : 1;
  const int offset = dim == 1 and tdim > 2
                         ? 3 * mesh::cell_num_entities(topology.cell_type(), 2)
                         : 0;
  REQUIRE(perms.size() == info.size() * num_entities);
  for (std::size_t c = 0; c < info.size(); ++c)
    for (int i = 0; i < num_entities; ++i)
      CHECK(perms[c * num_entities + i]
            == ((info[c] >> (offset + bits * i)) & ((1 << bits) - 1)));

  CHECK(any_nonzero(perms));
}

/// Check every dimension that has an orientation, i.e. all but the
/// vertices and the cell itself
void check_mesh(const mesh::Mesh<double>& mesh)
{
  mesh::Topology& topology = *mesh.topology_mutable();
  for (int dim = 1; dim < topology.dim(); ++dim)
    check_dimension(topology, dim);
}
} // namespace

TEST_CASE("Entity permutations match the packed cell permutations",
          "[mesh][permutations]")
{
  // Topology::create_entity_permutations unpacks the permutations from
  // the packed cell permutations, which are computed from the cell
  // vertices alone. mesh::compute_entity_permutations instead measures
  // the entities of one dimension against the cells incident to them.
  // The two must agree, and only this test compares them.
  constexpr std::array<std::array<double, 2>, 2> rectangle = {{{0, 0}, {1, 1}}};
  constexpr std::array<std::array<double, 3>, 2> box = {{{0, 0, 0}, {1, 1, 1}}};

  SECTION("triangle")
  {
    check_mesh(mesh::create_rectangle<double>(MPI_COMM_WORLD, rectangle, {3, 3},
                                              mesh::CellType::triangle));
  }
  SECTION("quadrilateral")
  {
    check_mesh(mesh::create_rectangle<double>(MPI_COMM_WORLD, rectangle, {3, 3},
                                              mesh::CellType::quadrilateral));
  }
  SECTION("tetrahedron")
  {
    check_mesh(mesh::create_box<double>(MPI_COMM_WORLD, box, {2, 2, 2},
                                        mesh::CellType::tetrahedron));
  }
  SECTION("hexahedron")
  {
    check_mesh(mesh::create_box<double>(MPI_COMM_WORLD, box, {2, 2, 2},
                                        mesh::CellType::hexahedron));
  }
  SECTION("prism")
  {
    // The only cell type with two facet types, where the two routes
    // differ most: one indexes faces by type, the other by cell-local
    // index
    check_mesh(mesh::create_box<double>(MPI_COMM_WORLD, box, {2, 2, 2},
                                        mesh::CellType::prism));
  }
}
