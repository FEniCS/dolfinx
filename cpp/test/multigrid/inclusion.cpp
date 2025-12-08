// Copyright (C) 2024 Paul T. Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include <mpi.h>

#include <catch2/catch_approx.hpp>
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

TEMPLATE_TEST_CASE("Gather global", "[multigrid]", double, float)
{
  using T = TestType;
  MPI_Comm comm = MPI_COMM_WORLD;
  auto comm_size = dolfinx::MPI::size(comm);
  auto rank = dolfinx::MPI::rank(comm);
  std::vector<T> local{static_cast<T>(rank), static_cast<T>(rank + 1)};

  std::vector<T> global = multigrid::gather_global<T>(
      std::span(local.data(), local.size()), comm_size * 2, comm);

  CHECK(global.size() == static_cast<std::size_t>(2 * comm_size));
  for (int i = 0; i < comm_size; i++)
  {
    CHECK(global[2 * i] == Catch::Approx(i));
    CHECK(global[2 * i + 1] == Catch::Approx(i + 1));
  }
}

template <std::floating_point T>
void CHECK_inclusion_map(const dolfinx::mesh::Mesh<T>& from,
                         const dolfinx::mesh::Mesh<T>& to,
                         const std::vector<std::int64_t>& map)
{
  const common::IndexMap& im_from = *from.topology()->index_map(0);
  const common::IndexMap& im_to = *to.topology()->index_map(0);

  std::vector<T> global_x_to = multigrid::gather_global(
      to.geometry().x().subspan(0, im_to.size_local() * 3),
      im_to.size_global() * 3, to.comm());

  REQUIRE(static_cast<std::int64_t>(map.size())
          == im_from.size_local() + im_from.num_ghosts());
  for (std::int64_t i = 0; i < static_cast<std::int64_t>(map.size()); i++)
  {
    CHECK(std::abs(from.geometry().x()[3 * i] - global_x_to[3 * map[i]])
          < std::numeric_limits<T>::epsilon());
    CHECK(std::abs(from.geometry().x()[3 * i + 1] - global_x_to[3 * map[i] + 1])
          < std::numeric_limits<T>::epsilon());
    CHECK(std::abs(from.geometry().x()[3 * i + 2] - global_x_to[3 * map[i] + 2])
          < std::numeric_limits<T>::epsilon());
  }
}

/// Performs one uniform refinement and checks the inclusion map between coarse
/// and fine mesh.
template <std::floating_point T>
void TEST_inclusion(dolfinx::mesh::Mesh<T>&& mesh_coarse)
{
  mesh_coarse.topology()->create_entities(1);

  auto [mesh_fine, parent_cell, parent_facet]
      = refinement::refine(mesh_coarse, std::nullopt);
  mesh_fine.topology()->create_connectivity(1, 0);
  mesh_fine.topology()->create_connectivity(0, 1);
  std::vector<std::int64_t> inclusion_map
      = multigrid::inclusion_mapping(mesh_coarse, mesh_fine, true);

  CHECK_inclusion_map(mesh_coarse, mesh_fine, inclusion_map);
}

TEMPLATE_TEST_CASE("Inclusion (interval)", "[multigrid][inclusion]", double,
                   float)
{
  for (auto n : {10})
  {
    TEST_inclusion(dolfinx::mesh::create_interval<TestType>(MPI_COMM_WORLD, n,
                                                            {0.0, 1.0}));
  }
}

TEMPLATE_TEST_CASE("Inclusion (triangle)", "[multigrid][inclusion]", double,
                   float)
{
  for (auto n : {5})
  {
    TEST_inclusion(dolfinx::mesh::create_rectangle<TestType>(
        MPI_COMM_WORLD, {{{0, 0}, {1, 1}}}, {n, n}, mesh::CellType::triangle));
  }
}

TEMPLATE_TEST_CASE("Inclusion (tetrahedron)", "[multigrid][inclusion]", double,
                   float)
{
  for (auto n : {5})
  {
    TEST_inclusion(dolfinx::mesh::create_box<TestType>(
        MPI_COMM_WORLD, {{{0, 0, 0}, {1, 1, 1}}}, {n, n, n},
        mesh::CellType::tetrahedron));
  }
}
