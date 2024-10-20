// Copyright (C) 2024 Paul T. Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <array>
#include <concepts>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include <mpi.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <basix/cell.h>

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx/multigrid/inclusion.h>
#include <dolfinx/refinement/refine.h>

using namespace dolfinx;
using namespace Catch::Matchers;

template <std::floating_point T>
void CHECK_inclusion_map(const dolfinx::mesh::Mesh<T>& from,
                         const dolfinx::mesh::Mesh<T>& to,
                         const std::vector<std::int64_t>& map)
{
  const common::IndexMap& im_from = *from.topology()->index_map(0);
  const common::IndexMap& im_to = *to.topology()->index_map(0);

  // 1) exchange local sizes
  std::vector<std::int32_t> local_sizes(dolfinx::MPI::size(from.comm()));
  {
    std::array<std::int32_t, 1> tmp{im_to.size_local() * 3};
    MPI_Allgather(&tmp, 1, MPI_INT32_T, local_sizes.data(), 1, MPI_INT32_T,
                  from.comm());
  }

  // 2) compute displacement vector
  std::vector<std::int32_t> displacements(local_sizes.size() + 1, 0);
  std::partial_sum(local_sizes.begin(), local_sizes.end(),
                   displacements.begin() + 1);

  // 3) Allgather global x vector
  std::vector<T> global_x_to(im_to.size_global() * 3);
  MPI_Allgatherv(to.geometry().x().data(), im_to.size_local() * 3,
                 dolfinx::MPI::mpi_t<T>, global_x_to.data(), local_sizes.data(),
                 displacements.data(), dolfinx::MPI::mpi_t<T>, from.comm());

  REQUIRE(static_cast<std::int64_t>(map.size()) == im_from.size_global());
  for (std::int64_t i = 0; i < map.size(); i++)
  {
    std::array<std::int32_t, 1> local{-1};
    im_from.global_to_local(std::array<std::int64_t, 1>{i}, local);
    if (local[0] == -1)
      continue;

    CHECK(std::abs(from.geometry().x()[3 * local[0]] - global_x_to[3 * map[i]])
          < std::numeric_limits<T>::epsilon());
    CHECK(std::abs(from.geometry().x()[3 * local[0] + 1]
                   - global_x_to[3 * map[i] + 1])
          < std::numeric_limits<T>::epsilon());
    CHECK(std::abs(from.geometry().x()[3 * local[0] + 2]
                   - global_x_to[3 * map[i] + 2])
          < std::numeric_limits<T>::epsilon());
  }
}

/// Performs one uniform refinement and checks the inclusion map between coarse
/// and fine mesh against the (provided) list.
template <std::floating_point T>
void TEST_inclusion(dolfinx::mesh::Mesh<T>&& mesh_coarse)
{
  mesh_coarse.topology()->create_entities(1);

  auto [mesh_fine, parent_cell, parent_facet]
      = refinement::refine(mesh_coarse, std::nullopt);
  mesh_fine.topology()->create_connectivity(1, 0);
  mesh_fine.topology()->create_connectivity(0, 1);
  std::vector<std::int64_t> inclusion_map
      = multigrid::inclusion_mapping(mesh_coarse, mesh_fine);

  CHECK_inclusion_map(mesh_coarse, mesh_fine, inclusion_map);
}

TEMPLATE_TEST_CASE("Inclusion (interval)", "[multigrid][inclusion]", double,
                   float)
{
  TEST_inclusion(
      dolfinx::mesh::create_interval<TestType>(MPI_COMM_WORLD, 2, {0.0, 1.0}));
}

TEMPLATE_TEST_CASE("Inclusion (triangle)", "[multigrid][inclusion]", double,
                   float)
{
  TEST_inclusion(dolfinx::mesh::create_rectangle<TestType>(
      MPI_COMM_WORLD, {{{0, 0}, {1, 1}}}, {1, 1}, mesh::CellType::triangle));
}

// TODO: fix!
// TEMPLATE_TEST_CASE("Inclusion (tetrahedron)", "[multigrid][inclusion]",
// double,
//                    float)
// {
//   TEST_inclusion(dolfinx::mesh::create_box<TestType>(
//       MPI_COMM_WORLD, {{{0, 0, 0}, {1, 1, 1}}}, {1, 1, 1},
//       mesh::CellType::tetrahedron));
// }
